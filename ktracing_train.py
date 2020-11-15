import random
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
import warnings
from ktracing_utils import *
import json

warnings.filterwarnings(action='ignore')
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['PYTHONHASHSEED'] = str(CFG.seed)
random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
torch.cuda.manual_seed(CFG.seed)
torch.backends.cudnn.deterministic = True


settings = json.load(open('SETTINGS.json'))
parameters = read_yml('parameters.yaml')

time_str = time.strftime("%m%d-%H%M")
# convert_feather(settings)

logging.basicConfig(filename=os.path.join(settings['LOGS_DIR'], f'log_{time_str}.txt'),
                    level=logging.INFO, format="%(message)s")


def main():

    train_df, train_samples, cate_offset = \
        preprocess(settings=settings, parameters=parameters, mode='train', update_flag=True)

    CFG.cate_cols = parameters['cate_cols']
    CFG.cont_cols = parameters['cont_cols']
    CFG.total_cate_size = cate_offset

    model = encoders[CFG.encoder](CFG)
    model.cuda()
    model._dropout = CFG.dropout

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'CFG: {CFG.__dict__}')
    logging.info(f'CFG: {CFG.__dict__}')
    logging.info(f'parameters: {count_parameters(model)}')

    train_db = KTDataset(CFG, train_df, train_samples, aug=CFG.aug)

    train_loader = DataLoader(
        train_db, batch_size=CFG.batch_size, shuffle=False,
        num_workers=0, pin_memory=True)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=CFG.learning_rate,
                      weight_decay=CFG.weight_decay,
                      )

    num_train_optimization_steps = int(
        len(train_db) / CFG.batch_size / CFG.gradient_accumulation_steps) * 7
    logging.info(f'num_train_optimization_steps: {num_train_optimization_steps}')
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CFG.warmup_steps,
                                                num_training_steps=num_train_optimization_steps
                                                )

    def get_lr():
        return scheduler.get_lr()[0]

    log_df = pd.DataFrame(columns=(['EPOCH'] + ['LR'] + ['TRAIN_LOSS', 'auc']))

    CFG.input_filename = settings['TRAIN_PT']
    file_name = generate_file_name(CFG)
    curr_lr = get_lr()

    for epoch in range(CFG.start_epoch, CFG.num_train_epochs):
        model_file_name = f"{file_name}_epoch-{epoch}.pt"
        print('model file name:', model_file_name)
        train_loss, auc = train(train_loader, model, optimizer, epoch, scheduler)
        if epoch % CFG.test_freq == 0 and epoch >= 0:
            log_row = {'EPOCH': epoch, 'LR': curr_lr, 'TRAIN_LOSS': train_loss, 'auc': auc}
            log_df = log_df.append(pd.DataFrame(log_row, index=[0]), sort=False)
            print(log_row)
            logging.info(log_df.tail(20))

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the cust_model it-self

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'transformer',
            'state_dict': model_to_save.state_dict(),
            'log': log_df,
        },
            settings['MODEL_DIR'], model_file_name,
        )

        run_validation(settings=settings, parameters=parameters, CFG=CFG, model_name=model_file_name)
    print('done')


if __name__ == '__main__':
    main()
