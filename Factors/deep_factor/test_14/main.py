import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import optuna
from Exp import Exp
import sys
import os
import numpy as np

model_name = sys.argv[1]
ins_name = sys.argv[2]
exp_idx = sys.argv[3]
regular = True if sys.argv[4]=='l1' else False
device = sys.argv[5]

dir_x = r'/home/largefile/public/zjwang/HF_factors/f12/output_factors/lastday'
dir_y = r'/home/largefile/public/zjwang/1minbar_rtn/1minbar_240'
dir_f = r'/home/largefile/zjwang/filter_data'
dir_y_p = r'/home/largefile/zjwang/nn_factor/test_14/results'

def objective(trial):
    lr = trial.suggest_categorical('lr', [1e-4])
    epoches = trial.suggest_categorical('epoches', [10])
    batch_size = trial.suggest_categorical('batch_size', [128*10])
    back_interval = trial.suggest_categorical('back_interval', [480])
    short_interval = trial.suggest_categorical('short_interval', [30])
    interval_feature = trial.suggest_categorical('interval_feature', [10])


    params = {
        'date_len': 90,
        'back_days': 3,
        'interval': 5,
        'idx':14,
        'train_data_frac': 0.6,
        'back_interval': back_interval,
        'short_interval': short_interval,
        'interval_feature': interval_feature,
        'model': model_name,
        'lr': lr,
        'regular': regular,
        'alpha': 1e-4,
        'device': device,
        'dir_x': dir_x, 
        'dir_y' : dir_y,
        'dir_f' : dir_f,
        'dir_y_p' : dir_y_p,
        'is_filter': True,
        'ins': ins_name,
        'epoches': epoches,
        'freeze': True,
        'lr_decay': True,
        'trans_epoch': int(0.7*epoches)+1,
        'single_num': 20,
        'print_mode': False,
        'print_num': 100,
        'y_col_name': 'extra_return',
        'batch_size': batch_size,
        'exp_idx': exp_idx,
        'k': True
    }

    if model_name == 'mlp':
        params_net = {
                'dims': [1024, 512, 256],
                'output_dim': 1,
                'short_interval': short_interval
            }


    if model_name == 'lstm':
        lstm_num = trial.suggest_categorical('lstm_num', [2])
        attention = trial.suggest_categorical('attention', [True])
        params_net = {
            'lstm_dim': 50,
            'lstm_num': lstm_num,
            'output_dim': 1,
            'nhead': 7,
            'attention': attention,
            'short_interval': short_interval
            }
    
    print('############ --- params --- ############')
    print(params)
    print('############ --- net params --- ############')
    print(params_net)
    e = Exp(params, params_net)
    res, _, ic_mean = e.exp()
    id = trial._trial_id
    torch.save(e.net.state_dict(), os.path.join(e.output_dir, 'model.pkl'))
    np.savetxt(os.path.join(e.output_dir, 'ic.csv'), res)

    return ic_mean

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1)

print('\n ############ --- Best trial --- ############')
trial = study.best_trial

print('Trial value.', trial.value)
print('Params:')
for k, v in trial.params.items():
    print('   {}: {}'.format(k, v))

