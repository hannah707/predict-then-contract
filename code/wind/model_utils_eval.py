import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter

from regret_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BiasedLoss(nn.Module):
    def __init__(self, w_neg=1, w_pos=1):
        super(BiasedLoss, self).__init__()
        self.w_neg = w_neg
        self.w_pos = w_pos

    def forward(self, prediction, target):
        err = prediction - target
        loss = torch.where(err > 0, self.w_pos*(err**2), self.w_neg*(err**2))
        return loss.mean() 

loss_fn = nn.MSELoss()


def test_model(models,tasks,data_loader,count_q,model_key,use_forecast=False,model_type='mlp',load_bbm=False, ckpt_path='saved_models', robust=False, yield_bias=False, w_neg=1, w_pos=1, debug_regret=False):
    if yield_bias:
        assert w_neg!=w_pos
        loss_b = BiasedLoss(w_neg=w_neg, w_pos=w_pos)

    for model in models:
        model.eval()
    
    if '2h' in model_key and 'basegrid0' in model_key:
        model_key = model_key.replace('2h','2,')  

    if load_bbm:
        for (ii,model),task in zip(enumerate(models),tasks):
            if not (ii==1 and use_forecast):
                state_dict = torch.load(f'{ckpt_path}/best-best-model_{task}_{model_key}.pth')
                model.load_state_dict(state_dict)
                model.to(device)

    loss_acc = [0.0]*2
    total_regret = 0.0
    
    with torch.no_grad():
        for batch in data_loader:
            if use_forecast:
                inputs, pcReal, prReal, yieldReal, yieldForecast = batch
            else:
                inputs, pcReal, prReal, yieldReal, _ = batch

            if model_type=='rnn':
                input_dim = inputs.shape[0]
                inputs = inputs.unsqueeze(1) 
                inputs = inputs.expand(-1, input_dim, -1)              

            preds = []
            for (ii,model), label in zip(enumerate(models), [prReal, yieldReal]):
                pred = model(inputs)
                if ii==1 and yield_bias:
                    loss = loss_b(pred, label)*len(pred)
                else:
                    loss = loss_fn(pred, label)*len(pred)
                loss_acc[ii]+=loss.item()
                preds.append(pred)
            
            if count_q:
                [prPred, yieldPred] = preds

                if use_forecast:
                    Q_loss_batch, _ = Q_loss(prPred, yieldForecast, pcReal, prReal, yieldReal, robust=robust, debug_regret=debug_regret)
                else:
                    Q_loss_batch, _ = Q_loss(prPred, yieldPred, pcReal, prReal, yieldReal, robust=robust, debug_regret=debug_regret)
                total_regret +=  Q_loss_batch.item()

    return loss_acc, total_regret


def save_models(models,tasks,val_acc,val_regret, best_loss, best_loss_actual,patience, patience_alpha, epoch, counter, continue_training, model_key, use_forecast=False, val_method='acc', ckpt_path='saved_models'):
    assert val_method=='acc' or val_method=='reg', 'Wrong validation method keyword.'

    if val_method=='reg':
        if val_regret < best_loss*patience_alpha:
            best_loss = min([val_regret, best_loss])
            counter = 0
            for (ii,task),model in zip(enumerate([]),models):
                if not (ii==1 and use_forecast):
                    torch.save(model.state_dict(), f'{ckpt_path}/best-model_{task}_{model_key}.pth')
            if val_regret < best_loss_actual:
                best_loss_actual = val_regret
                for (ii,task),model in zip(enumerate(tasks),models):
                    if not (ii==1 and use_forecast):
                        torch.save(model.state_dict(), f'{ckpt_path}/best-best-model_{task}_{model_key}.pth')
        else:
            counter += 1
            print(f'Early-stopping counts: {counter}/{patience}')
            if counter >= patience:
                print(f'Validation regret did not improve for {patience} epochs. Stopping early at {epoch}.')
                continue_training = [False]*len(tasks)

    else:
        for (ii,task),model in zip(enumerate(tasks),models):
            if use_forecast:
                continue_training[1] = False
            if continue_training[ii]:
                loss = val_acc[ii]
                if loss < best_loss[ii]*patience_alpha:
                    best_loss[ii] = min([loss,best_loss[ii]])
                    counter[ii] = 0
                    torch.save(model.state_dict(), f'{ckpt_path}/best-model_{task}_{model_key}.pth')
                    if loss < best_loss_actual[ii]:
                        best_loss_actual[ii] = loss
                        torch.save(model.state_dict(), f'{ckpt_path}/best-best-model_{task}_{model_key}.pth')
                else:
                    counter[ii]+=1
                    print(f'Early-stopping counts: model-{ii}: {counter[ii]}/{patience}')
                    if counter[ii] >= patience:
                        print(f'Validation loss did not improve for {patience} epochs for {task}. Stopping early at {epoch}.')
                        continue_training[ii] = False
                        print('Current training status:', continue_training)

    return best_loss, best_loss_actual, counter, continue_training


def save_training_summary(train_losses, train_objs, val_losses, val_objs, model_key, prog_path='../../saved_progresses'):
    for obj,name in zip([train_losses, train_objs, val_losses, val_objs],['train-losses', 'train-objs', 'val-losses', 'val-objs']):
        np.save(os.path.join(prog_path,f'{name}_{model_key}.npy'), np.array(obj))

    plt.subplots(1,2,figsize=(9,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label=[i+' train loss' for i in ['rt-price','yield']])
    plt.plot(val_losses, label=[i+' val loss' for i in ['rt-price','yield']])
    plt.legend(loc='upper right')

    plt.subplot(1,2,2)
    plt.plot(train_objs, label='train regret')
    plt.plot(val_objs, label='val regret')
    plt.legend(loc='upper right')

    plt.tight_layout()


def test_model_me(models,tasks,data_loader,count_q,model_key,model_type='mlp',load_bbm=True, ckpt_path='saved_models', robust=False, debug_regret=False):
    for model in models:
        model.eval()
    
    loss_acc = [0.0]*2

    if 'sp' in model_key:
        for (ii,model),task in zip(enumerate(models),tasks):

            state_dict = torch.load(f'{ckpt_path}/best-model-before-switch_{task}_{model_key}.pth')
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            
        for batch in data_loader:
            inputs, pcReal, prReal, yieldReal, _ = batch

            if model_type=='rnn':
                input_dim = inputs.shape[0]
                inputs = inputs.unsqueeze(1) 
                inputs = inputs.expand(-1, input_dim, -1)              

            preds = []
            for (ii,model), label in zip(enumerate(models), [prReal, yieldReal]):
                pred = model(inputs)
                loss = loss_fn(pred, label)*len(pred)
                loss_acc[ii]+=loss.item()
                preds.append(pred)

        print(f'SUM test accuracy before switch {model_key} = {np.array(loss_acc).round(3)}')

    if load_bbm:
        for (ii,model),task in zip(enumerate(models),tasks):
            state_dict = torch.load(f'{ckpt_path}/best-best-model_{task}_{model_key}.pth')
            model.load_state_dict(state_dict, strict=False)
            model.to(device)

    loss_acc = [0.0]*2
    total_regret = 0.0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, pcReal, prReal, yieldReal, _ = batch

            if model_type=='rnn':
                input_dim = inputs.shape[0]
                inputs = inputs.unsqueeze(1) 
                inputs = inputs.expand(-1, input_dim, -1)              

            preds = []
            for (ii,model), label in zip(enumerate(models), [prReal, yieldReal]):
                pred = model(inputs)
                loss = (pred - label).sum()
                # loss = loss_fn(pred, label)*len(pred)
                loss_acc[ii]+=loss.item()
                # preds.append(pred)
            
            if count_q:
                [prPred, yieldPred] = preds

                Q_loss_batch, _ = Q_loss(prPred, yieldPred, pcReal, prReal, yieldReal, robust=robust, debug_regret=debug_regret)
                total_regret +=  Q_loss_batch.item()
                
    return loss_acc, total_regret
