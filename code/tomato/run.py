import os
import time
import datetime
import argparse
import numpy as np
import random

from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
# from torch.utils.tensorboard import SummaryWriter

from data import *
from models import *
from model_utils_tomato import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['CUDNN_DETERMINISTIC'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print('Using device:', device, torch.cuda.get_device_name(0))
else:
    print('Using device:', device)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # ==================== Parser setting ==========================
    parser = argparse.ArgumentParser(description='Tomato Day-Ahead Optimization')

    parser.add_argument('--seed', type=int, default=5, help='main seed')
    parser.add_argument('--npseed', type=int, default=23, help='np/random seed')
    parser.add_argument('--data', type=str, default='multivariate', help='dataset key')
    parser.add_argument('--method', type=str, default='base', help='keyword about method')
    parser.add_argument('--exp_key', type=str, default='test', help='keyword about experiment')
    parser.add_argument('--hidden_dim1', type=int, default=128, help='neurons in first layer')
    parser.add_argument('--hidden_dim2', type=int, default=256, help='neurons in last layer')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--patience', type=int, default=21, help='early stopping patience')
    parser.add_argument('--patience_alpha', type=float, default=1, help='if > 1, the model will use loss*alpha to count the early stopping patience')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='proportion of training set')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='proportion of validation set')
    parser.add_argument('--reg_price', type=float, default=0, help='ratio of SPO:MSE for training price prediction model')
    parser.add_argument('--reg_yield', type=float, default=0, help='ratio of SPO:MSE for training yield prediction model')
    parser.add_argument('--spoplus', type=str2bool, default=False, help='whether SPO+ loss is used in (price) prediction')
    parser.add_argument('--robust', type=str2bool, default=False, help='whether the decision-making uses robust optimization')
    parser.add_argument('--val_method', type=str, default='acc', help='validation loss for early stopping, either \'mse\' or \'reg')
    parser.add_argument('--lr_decay', type=float, default=1, help='learning rate change after warm up')
    parser.add_argument('--switch_patience', type=int, default=0, help='epochs of warming up, counted after converging')
    parser.add_argument('--switch_by_patience', type=str2bool, default=True, help='if False, the warm up uses fixed epochs')
    parser.add_argument('--single_warm_up', type=str, default='', help='model name if just one model will be warmed up')
    parser.add_argument('--load_model', type=str, default='', help='keyword of model checkpoints to load')
    parser.add_argument('--start_epoch', type=int, default=0, help='the number of the starting epoch')
    parser.add_argument('--data_path', type=str, default='../../inputs/tomato/', help='path to input data')
    parser.add_argument('--ckpt_path', type=str, default='../../saved_models/', help='path to model checkpoints')
    parser.add_argument('--take_subset', type=int, default=1, help='take every xth of the data')
    parser.add_argument('--save_prog', type=str2bool, default=True, help='saving progress into npy file')
    parser.add_argument('--save_after_switch', type=str2bool, default=False, help='save every model after warming up')
    parser.add_argument('--save_cycle', type=int, default=10, help='save progress every xth epochs')
    parser.add_argument('--yield_bias', type=str2bool, default=False, help='if yield prediction model is trained with a weighted MSE')
    parser.add_argument('--w_neg', type=float, default=1, help='weight of under-estimation')
    parser.add_argument('--w_pos', type=float, default=1, help='weight of over-estimation')
    parser.add_argument('--debug_regret', type=str2bool, default=False, help='whether to show errored values for cvxpy')

    args = parser.parse_args()

    SEED = args.seed
    def worker_init_fn(worker_id):
        torch.manual_seed(SEED + worker_id)
    npseed = args.npseed
    random.seed(npseed)
    np.random.seed(npseed)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rtPrice = np.load(os.path.join(args.data_path,'realtime-prices.npy'))
    ctPrice = np.load(os.path.join(args.data_path,'contract-prices.npy'))
    features = np.load(os.path.join(args.data_path,'multivariate-features.npy'))
    yl_act = np.load(os.path.join(args.data_path,'yields.npy'))

    features = np.hstack((features, ctPrice.reshape(-1,1)))
    assert len(features) == len(ctPrice) == len(rtPrice) == len(yl_act), 'length of features mismatch!'
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    input_dim = features.shape[1]
    
    if args.take_subset>1:
        rtPrice = rtPrice[::args.take_subset]
        ctPrice = ctPrice[::args.take_subset]
        features = features[::args.take_subset]
        yl_act = yl_act[::args.take_subset]

    dataset = TomatoPriceDataset(features, ctPrice, rtPrice, yl_act)
    train_size = int(args.train_ratio * len(dataset))
    val_size = int(args.val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print('Train-val-test:', len(train_dataset), len(val_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, worker_init_fn=worker_init_fn)

    tasks = ['realtime-Price','yield']

    REG = [i if i < 1000 else np.inf for i in [args.reg_price, args.reg_yield]]

    model_p = mlpRegressor3(input_dim, args.hidden_dim1, args.hidden_dim2, output_dim=1)
    model_p.to(device)

    model_y = mlpRegressor3(input_dim, args.hidden_dim1, args.hidden_dim2, output_dim=1)
    model_y.to(device)

    if len(args.load_model)>0:
        load_model = args.load_model
        model_timestamp = load_model[19:34]
        for (ii,model),task in zip(enumerate([model_p,model_y]),tasks):
            if os.path.exists(os.path.exists(args.model_path,f'best-model_{task}_{load_model}.pth')): # latest saving
                state_dict = torch.load(os.path.exists(args.model_path,f'best-model_{task}_{load_model}.pth'))
            elif os.path.exists(os.path.exists(args.model_path,f'./saved_models/best-best-model_{task}_{load_model}.pth')): # best saving
                state_dict = torch.load(os.path.exists(args.model_path,f'./saved_models/best-best-model_{task}_{load_model}.pth'))
            else:
                print('No model found, start from scratch.')
                model_timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                break
            model.load_state_dict(state_dict)
            model.to(device)
    else:
        model_timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    start_time = time.time()

    train_params = {'SEED','num_epochs','lr','val_method','patience','patience_alpha','switch_patience','switch_by_patience','single_warm_up','start_epoch','ckpt_path','save_prog','save_after_switch','exp_key','method','data','spoplus','robust','yield_bias', 'w_neg', 'w_pos'}
    train_args = argparse.Namespace(**{k: v for k, v in vars(args).items() if k in train_params})
    train_losses, train_objs, val_losses, val_objs = train_models([model_p,model_y], train_loader, val_loader=val_loader, test_loader=test_loader, extra_key=model_timestamp, REG=REG, device=device, **vars(train_args))

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_seconds = int(elapsed_time)
    elapsed_time_hours = np.round(elapsed_time/3600,2)
    print(f"Total training time: {elapsed_time_seconds} seconds, {elapsed_time_hours} hours.")