import random
import numpy as np
import warnings
import json
from ktracing_utils import *
import time
warnings.filterwarnings(action='ignore')

os.environ['OMP_NUM_THREADS'] = '16'
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['PYTHONHASHSEED'] = str(CFG.seed)

settings = json.load(open('SETTINGS.json'))
parameters = read_yml('parameters.yaml')
for key, value in parameters.items():
    setattr(CFG, key, value)

def main():

    results_u_path = os.path.join(settings["CLEAN_DATA_DIR"], 'user_dict.pkl')
    start = time.time()
    if not os.path.isfile(results_u_path):

        file_name = "train_v0.feather"
        df_ = feather.read_dataframe(os.path.join(settings['RAW_DATA_DIR'], file_name))
        df_ = df_.groupby('user_id').tail(CFG.window_size)
        df_, _, _ = preprocess_data(df_, parameters=parameters, settings=settings)
        features = CFG.cate_cols + CFG.cont_cols + [TARGET]
        df_ = df_[['user_id'] + features]
        user_dict = {uid: u.values[:, 1:] for uid, u in df_.groupby('user_id')}
        with open(results_u_path, 'wb') as handle:
            pickle.dump(user_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(results_u_path, 'rb') as handle:
            user_dict = pickle.load(handle)
    print(f'process time: {time.time() - start} seconds')
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    torch.cuda.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True
    model_file_name = 'b-128_a-TRANSFORMER_e-20_h-20_d-0.2_l-2_hd-10_s-123_len-20_aug-0.0_da-trainsamplev1_epoch-0.pt'
    CFG.batch_size = 128
    CFG.features = CFG.cate_cols + CFG.cont_cols + [TARGET]
    settings['VALIDATION_DATASET'] = 'validation_sample_v1.feather'

    file_name = settings['VALIDATION_DATASET']
    valid_df = feather.read_dataframe(os.path.join(settings['RAW_DATA_DIR'], file_name))

    run_validation(valid_df, settings=settings, parameters=parameters, CFG=CFG,
                   model_name=model_file_name, user_dict=user_dict)

    df_sample = pd.read_csv(os.path.join(settings['RAW_DATA_DIR'], 'example_test.csv'))
    #
    df_sample[TARGET] = 0.5
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
    for test_batch in sample_batch:
        i += 1
        # update state
        if df_batch_prior is not None:
            answers = eval(test_batch['prior_group_answers_correct'].iloc[0])
            df_batch_prior['answered_correctly'] = answers
            print('auc:', metrics.roc_auc_score(answers, predictions))

            print('comparison', pd.DataFrame({'ACT': answers, 'PRED': predictions.tolist()}))

        # save prior batch for state update

        test_loader, test_df, _ = get_dataloader(test_batch, settings, parameters, CFG,
                                                 user_dict=user_dict, prior_df=df_batch_prior)
        df_batch_prior = test_df[['user_id'] + CFG.features]
        predictions = run_test(test_loader, settings=settings, CFG=CFG, model_name=model_file_name)

        # get state
        df_batch = test_batch[test_batch.content_type_id == 0]
        df_batch['answered_correctly'] = predictions


if __name__ == '__main__':
    main()
