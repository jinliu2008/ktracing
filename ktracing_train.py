import random
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
import warnings
from ktracing_utils import *
import json
import sys
import psutil
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

    print(f'CFG: {CFG.__dict__}')
    logging.info(f'CFG: {CFG.__dict__}')

    if submission_flag:
        file_path = settings["SUBMISSION_DIR"]
    else:
        file_path = settings["CLEAN_DATA_DIR"]
    mappers_dict_path = os.path.join(file_path, 'mappers_dict.pkl')
    with open(mappers_dict_path, 'rb') as handle:
        mappers_dict = pickle.load(handle)


    file_name = settings['TRAIN_DATASET']
    df_ = feather.read_dataframe(os.path.join(settings['RAW_DATA_DIR'], file_name))
    df_ = df_[df_.content_type_id==False]

    # arrange by timestamp
    df_ = df_.sort_values(['timestamp'], ascending=True).reset_index(drop=True)

    col = 'content_id'
    cate_offset =1
    cate2idx = mappers_dict[col]
    df_.loc[:, col] = df_[col].map(cate2idx).fillna(0).astype(int)
    cate_offset += len(cate2idx)

    skills = df_["content_id"].unique()
    sample_size = df_["user_id"].nunique()
    n_skill = cate_offset

    print("number skills", len(skills))
    group = df_[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(lambda r: (
        r['content_id'].values,
        r['answered_correctly'].values))
    del df_
    gc.collect()

    dataset = SAKTDataset(group, n_skill)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=8)

    item = dataset.__getitem__(5)
    model = encoders[CFG.encoder](n_skill, embed_dim=128)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(CFG.start_epoch, CFG.num_train_epochs):

        model_file_name = generate_file_name(CFG)
        model_file_name = f"{model_file_name}_epoch-{epoch}.pt"
        print('model file name:', model_file_name)

        loss, acc, auc = SAKT_train(model, dataloader, optimizer)
        # loss, acc, auc = SAKT_train(model, dataloader, n_skill, optimizer, epoch, scheduler)
        print("epoch - {} train_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(epoch, loss, acc, auc))
        #
        # if epoch % CFG.test_freq == 0 and epoch >= 0:
        #     log_row = {'EPOCH': epoch, 'TRAIN_LOSS': train_loss, 'auc': auc}
        #     log_df = log_df.append(pd.DataFrame(log_row, index=[0]), sort=False)
        #     print(log_row)
        #     logging.info(log_df.tail(20))
        #
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the cust_model it-self
        #
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': 'transformer',
        #     'state_dict': model_to_save.state_dict(),
        #     'log': log_df,
        # },
        #     settings['MODEL_DIR'], model_file_name,
        # )
        #
        #

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for epoch in range(CFG.start_epoch, CFG.num_train_epochs):


            df_sample = pd.read_csv(os.path.join(settings['RAW_DATA_DIR'], 'example_test.csv'))
            #
            df_sample[TARGET] = df_sample['content_type_id']
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
            answers_all = []
            predictions_all = []

            prev_test_df = None

            model.eval()

            for test_df in sample_batch:
                # HDKIM
                if (prev_test_df is not None) & (psutil.virtual_memory().percent < 90):
                    print(psutil.virtual_memory().percent)
                    answers = eval(test_df['prior_group_answers_correct'].iloc[0])
                    prev_test_df['answered_correctly'] = answers
                    answers_all += answers

                    predictions_all += outs

                    prev_test_df = prev_test_df[prev_test_df.content_type_id == False]
                    prev_group = prev_test_df[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(
                        lambda r: (
                            r['content_id'].values,
                            r['answered_correctly'].values))
                    for prev_user_id in prev_group.index:
                        prev_group_content = prev_group[prev_user_id][0]
                        prev_group_ac = prev_group[prev_user_id][1]
                        if prev_user_id in group.index:
                            group[prev_user_id] = (np.append(group[prev_user_id][0], prev_group_content),
                                                   np.append(group[prev_user_id][1], prev_group_ac))

                        else:
                            group[prev_user_id] = (prev_group_content, prev_group_ac)
                        if len(group[prev_user_id][0]) > MAX_SEQ:
                            new_group_content = group[prev_user_id][0][-MAX_SEQ:]
                            new_group_ac = group[prev_user_id][1][-MAX_SEQ:]
                            group[prev_user_id] = (new_group_content, new_group_ac)

                prev_test_df = test_df.copy()

                # HDKIMHDKIM

                test_df = test_df[test_df.content_type_id == False]

                test_dataset = TestDataset(group, test_df, skills)
                test_dataloader = DataLoader(test_dataset, batch_size=51200, shuffle=False)

                outs = []

                for item in tqdm(test_dataloader):
                    x = item[0].to(device).long()
                    target_id = item[1].to(device).long()

                    with torch.no_grad():
                        output, att_weight = model(x, target_id)

                    output = torch.sigmoid(output)
                    output = output[:, -1]

                    # pred = (output >= 0.5).long()
                    # loss = criterion(output, label)

                    # val_loss.append(loss.item())
                    # num_corrects += (pred == label).sum().item()
                    # num_total += len(label)

                    # labels.extend(label.squeeze(-1).data.cpu().numpy())
                    outs.extend(output.view(-1).data.cpu().numpy())

                test_df['answered_correctly'] = outs
                predictions = outs

            print('sample auc:', metrics.roc_auc_score(answers_all, predictions_all))

if __name__ == '__main__':
    main()
