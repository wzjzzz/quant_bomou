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
dir_y = r'/home/largefile/public/zjwang/1minbar_rtn/1minbar_30'
dir_f = r'/home/largefile/zjwang/filter_data'
dir_y_p = r'/home/largefile/zjwang/nn_factor/test_00/results'

def objective(trial):
    lr = trial.suggest_categorical('lr', [1e-4])
    epoches = trial.suggest_categorical('epoches', [16])
    batch_size = trial.suggest_categorical('batch_size', [128])
    back_interval = trial.suggest_categorical('back_interval', [30])
    interval_feature = trial.suggest_categorical('interval_feature', [1])


    params = {
        'date_len': 90,
        'back_days': 1,
        'interval': 0,
        'idx':14,
        'train_data_frac': 0.6,
        'back_interval': back_interval,
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
        'trans_epoch': 0.5*int(epoches)+1,
        'single_num': 20,
        'print_mode': False,
        'print_num': 100,
        'y_col_name': 'extra_return',
        'batch_size': batch_size,
        'exp_idx': exp_idx
    }

    if model_name == 'mlp':
        params_net = {
                'input_dim': back_interval//params['interval_feature'] * 50 ,   # back_interval *  (49 factors + 1 time)
                'dims': [1024, 512, 256],
                'output_dim': 1,
            }


    if model_name == 'lstm':
        lstm_num = trial.suggest_categorical('lstm_num', [2])
        attention = trial.suggest_categorical('attention', [True])
        params_net = {
            'input_dim': 50,   # 49 + 1 factors  序列长度：back_days 
            'lstm_dim': 50,
            'lstm_num': lstm_num,
            'output_dim': 1,
            'nhead': 5,
            'attention': attention
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

