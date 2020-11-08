import json
from ktracing_utils import read_yml, preprocess
pd.options.mode.chained_assignment = None


settings = json.load(open('SETTINGS.json'))
parameters = read_yml('parameters.yaml')


# data load
print('loading data')

train_samples, train_users = preprocess(settings=settings, parameters=parameters, mode='TRAIN')

# train
train_samples, train_users = preprocess(settings=settings, parameters=parameters, mode='TRAIN')
torch.save([train_samples, train_df, mappers_dict, cate_offset, cate_cols, cont_cols],
           'ktracing_train_v0.pt')