import random
import numpy as np
import warnings
import json
from ktracing_utils import *

warnings.filterwarnings(action='ignore')

os.environ['OMP_NUM_THREADS'] = '16'
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['PYTHONHASHSEED'] = str(CFG.seed)

settings = json.load(open('SETTINGS.json'))
parameters = read_yml('parameters.yaml')


def main():

    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    torch.cuda.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True
    path = 'b-1024_a-TRANSFORMER_e-20_h-20_d-0.2_l-2_hd-10_s-123_len-20_aug-0.0_da-trainsamplev1_epoch-0.pt'
    # settings['VALIDATION_DATASET']
    CFG.batch_size = 1024*64
    settings['VALIDATION_DATASET'] = 'train_sample_v1.feather'
    run_validation(settings=settings, parameters=parameters, CFG=CFG, model_name=path)


if __name__ == '__main__':
    main()
