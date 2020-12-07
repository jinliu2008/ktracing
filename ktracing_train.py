import random
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
import warnings
from ktracing_utils import *
import json

warnings.filterwarnings(action='ignore')

submission_flag = 1

def main():

    settings = json.load(open('SETTINGS.json'))
    parameters = read_yml('parameters.yaml')
    for key, value in parameters.items():
        setattr(CFG, key, value)
    time_str = time.strftime("%m%d-%H%M")

    logging.basicConfig(filename=os.path.join(settings['LOGS_DIR'], f'log_{time_str}.txt'),
                        level=logging.INFO, format="%(message)s")

    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['NUMEXPR_MAX_THREADS'] = '16'
    os.environ['PYTHONHASHSEED'] = str(CFG.seed)
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    torch.cuda.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True
    CFG.features = CFG.cate_cols + CFG.cont_cols + [TARGET]


    print(f'process time: {time.time() - start} seconds')
    print(f'CFG: {CFG.__dict__}')
    logging.info(f'CFG: {CFG.__dict__}')
    file_name = settings['TRAIN_DATASET']
    df_ = feather.read_dataframe(os.path.join(settings['RAW_DATA_DIR'], file_name))
    train_loader, _, sample_size = get_dataloader(df_, settings, parameters, CFG, user_dict={})
    model = encoders[CFG.encoder](CFG)
    model.cuda()
    model._dropout = CFG.dropout

    logging.info(f'parameters: {count_parameters(model)}')

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
        sample_size / CFG.batch_size / CFG.gradient_accumulation_steps) * 7
    logging.info(f'num_train_optimization_steps: {num_train_optimization_steps}')
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CFG.warmup_steps,
                                                num_training_steps=num_train_optimization_steps
                                                )

    log_df = pd.DataFrame(columns=(['EPOCH', 'TRAIN_LOSS', 'auc']))

    CFG.input_filename = file_name

    for epoch in range(CFG.start_epoch, CFG.num_train_epochs):
        model_file_name = generate_file_name(CFG)
        model_file_name = f"{model_file_name}_epoch-{epoch}.pt"
        print('model file name:', model_file_name)

        train_loss, auc = train(train_loader, model, optimizer, epoch, scheduler)

        if epoch % CFG.test_freq == 0 and epoch >= 0:
            log_row = {'EPOCH': epoch, 'TRAIN_LOSS': train_loss, 'auc': auc}
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

        if submission_flag == 1:
            results_u_path = os.path.join(settings["CLEAN_DATA_DIR"], 'user_dict.pkl')
        else:
            results_u_path = os.path.join(settings["CLEAN_DATA_DIR"], 'user_dict_v0.pkl')
        start = time.time()
        if not os.path.isfile(results_u_path):
            if submission_flag == 1:
                input_file_name = "train.feather"
            else:
                input_file_name = "train_v0.feather"
            print(f'input: {input_file_name}')
            df_ = feather.read_dataframe(os.path.join(settings['RAW_DATA_DIR'], input_file_name))
            df_ = df_.groupby('user_id').tail(CFG.window_size)
            df_, _, _ = preprocess_data(df_, parameters=parameters, settings=settings)

            df_ = df_[['user_id'] + CFG.features]
            user_dict = {uid: u.values[:, 1:] for uid, u in df_.groupby('user_id')}
            with open(results_u_path, 'wb') as handle:
                pickle.dump(user_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(results_u_path, 'rb') as handle:
                user_dict = pickle.load(handle)

        # file_name = settings['VALIDATION_DATASET']
        # valid_df = feather.read_dataframe(os.path.join(settings['RAW_DATA_DIR'], file_name))
        # run_validation(valid_df, settings=settings, parameters=parameters, CFG=CFG,
        #                model_name=model_file_name, user_dict=user_dict)

    for epoch in range(CFG.start_epoch, CFG.num_train_epochs):

        df_sample = pd.read_csv(os.path.join(settings['RAW_DATA_DIR'], 'example_test.csv'))
        #
        df_sample[TARGET] = -df_sample['content_type_id']
        sample_batch = []

        # batch 1
        sample_batch.append(df_sample.iloc[:18])
        # batch 2
        sample_batch.append(df_sample.iloc[18:45])
        # batch 3
        sample_batch.append(df_sample.iloc[45:71])
        # batch 4
        sample_batch.append(df_sample.iloc[71:])

        df_batch_prior = None
        i = 0
        answers_all = []
        predictions_all = []
        for test_batch in sample_batch:
            i += 1
            # update state
            if df_batch_prior is not None:
                answers = eval(test_batch['prior_group_answers_correct'].iloc[0])
                df_batch_prior['answered_correctly'] = answers
                answers_all += answers.copy()
                predictions_all += [p[0] for p in predictions.tolist()]

            # save prior batch for state update
            test_loader, test_df, _ = get_dataloader(test_batch, settings, parameters, CFG,
                                                     user_dict=user_dict, prior_df=df_batch_prior)
            df_batch_prior = test_df[['user_id'] + CFG.features]
            predictions = run_test(test_loader, settings=settings, CFG=CFG, model_name=model_file_name)

            # get state
            df_batch = test_batch[test_batch.content_type_id == 0]
            df_batch['answered_correctly'] = predictions

        print('sample auc:', metrics.roc_auc_score(answers_all, predictions_all))

if __name__ == '__main__':
    main()
