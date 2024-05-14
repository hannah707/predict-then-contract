#%%
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from regret_utils import *
from model_utils_eval import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_weights(REG):
    if REG[0]!=np.inf:
        w11 = 1/(1+REG[0])
        w12 = REG[0]/(1+REG[0])
    else:
        w11,w12 = 0.0, 1.0
    if REG[1]!=np.inf:
        w21 = 1/(1+REG[1])
        w22 = REG[1]/(1+REG[1])
    else:
        w21,w22 = 0.0, 1.0

    return w11,w12,w21,w22

def loss_adding(l1, l2):
    l1_ = l1.detach().clone()+l2.detach().clone()
    l1_.requires_grad = True
    return l1_

def train_models(models, train_loader,\
                tasks=['rp','yl'],  REG=[0,0], spoplus=False, robust=False, \
                val_loader=None, val_method='acc', test_loader=None, \
                num_epochs=10, lr=0.0001, lr_decay=1, patience=2, patience_alpha=1,  \
                switch_patience=0, switch_by_patience=True, single_warm_up='', save_after_switch=False, \
                start_epoch=0, ckpt_path='saved_models', save_from=np.inf, save_prog=True, \
                exp_key='latest', method='', data='', extra_key='', \
                yield_bias=False, w_neg=1, w_pos=1, \
                save_cycle=10, tb=False, seed=2023, device=device):                       
                        
    model_key = f'tomato_{data}_{method}_{exp_key}_{extra_key}'

    for model in models:
        model.to(device)
        
    loss_fn = nn.MSELoss()
    if yield_bias:
        assert w_neg!=w_pos
        loss_b = BiasedLoss(w_neg=w_neg, w_pos=w_pos)

    torch.manual_seed(seed)
    
    es_counter = np.inf
    switched = False
    
    # if backward_loss == 'regret': REG = [np.inf, np.inf]
    if switch_patience!=0: 
        REG_ = REG
        REG = [0,0]
        # backward_loss = 'accuracy'
        val_method = 'acc'
        count_q = False

    # if tb:
    #     writer = SummaryWriter()
    
    optimizerP = optim.SGD(models[0].parameters(), lr=lr, weight_decay=1e-5)
    optimizerY = optim.SGD(models[1].parameters(), lr=lr, weight_decay=1e-5)
    # if len(models)>2:
    #     optimizerC = optim.SGD(models[2].parameters(), lr=lr, weight_decay=1e-5)

    train_losses, train_objs = [],[] #Those are sum over training dataset
    assert single_warm_up=='' or single_warm_up=='yield' or single_warm_up=='price'
    if single_warm_up=='yield':
        assert len(models)==2
        continue_training = [False, True]
    elif single_warm_up=='price':
        assert len(models)==2
        continue_training = [True, False]
    else:
        continue_training = [True]*len(models)
    if val_loader is not None:
        val_losses, val_objs = [],[] #Those are sum over validation dataset
        es_counter = [0]*len(models)
        if REG==0 or REG==[0,0] or REG==(0,0):
            val_method = 'acc'
            print('Using accuracy as validation method in independent model training.')
        assert val_method=='acc' or val_method=='reg', 'Wrong validation method keyword.'
        if val_method =='acc':
            best_loss = [np.inf]*len(models)
            best_loss_actual = [np.inf]*len(models)
        else:
            best_loss = np.inf
            best_loss_actual = np.inf

    is_rnn_layer = [False]*len(models)

    if len(REG)==1:
        REG = [REG]*len(models)
    elif len(REG)!=len(models):
        REG = [0.0]*len(models)
        print('Reshape REG to match the model length:', REG)

    w11,w12,w21,w22 = loss_weights(REG)

    if w12+w22!=0 or val_method!='acc': #or backward_loss!='accuracy':
        count_q = True
    else:
        count_q = False

    if 'biasedmse' in method: count_q = False

    assert not count_q==False or spoplus==False , "spoplus should be False if count_q is False"

    print(f'Price prediction loss:{w11}*MSE+{w12}*SPO. Yield prediction loss:{w21}*MSE+{w22}*SPO.')

    for ii,model in enumerate(models):
        first_layer = next(model.children())
        is_rnn_layer[ii] = isinstance(first_layer, (nn.RNN, nn.LSTM, nn.GRU))
    
    print(f"Model:{model_key} starts training.")
    print("=============================================================")
        
    for epoch in range(start_epoch, num_epochs+1):
        # print(f'Epoch {epoch}, we train models', continue_training)
        for model, (ii, training) in zip(models, enumerate(continue_training)):
            if training:
                model.train()

        train_loss_acc = [0.0]*len(models)
        train_regret = 0.0
        all_errored_decisions = 0
        errored_decision = 0  

        for batch_data in train_loader:
            inputs, dpReal, rpReal, yieldReal = batch_data

            preds = []
            for ii,model in enumerate(models):
                if is_rnn_layer[ii]:
                    input_dim = inputs.shape[0]
                    inputs2 = inputs.unsqueeze(1) 
                    inputs2 = inputs2.expand(-1, input_dim, -1)
                    model_type = 'rnn' 
                    pred = model(inputs2)
                else:
                    model_type = 'mlp'
                    pred = model(inputs)
                preds.append(pred)

            if len(models) == 2:
                [rpPred, yieldPred] = preds
                if yield_bias:
                    acc_loss = [loss_fn(rpPred, rpReal), loss_b(yieldPred, yieldReal)]
                else:
                    acc_loss = [loss_fn(rpPred, rpReal), loss_fn(yieldPred, yieldReal)]
            else:
                raise ValueError('Invalid number of models.')
            
            Q_loss_batch = None
            errored_decision = 0       
            
            if count_q:
                Q_loss_batch, errored_decision = Q_loss(rpPred, yieldPred, dpReal, rpReal, yieldReal, robust=robust)

                if errored_decision == len(inputs):
                    print('No valid decisions are made for this batch in epoch ', epoch)

                train_regret += Q_loss_batch.clone().detach().cpu().item()
                Q_loss_batch = Q_loss_batch/len(inputs)

                if spoplus:
                    Q_loss_plus_batch, _ = Q_loss_spoplus(rpPred, yieldPred, dpReal, rpReal, yieldReal)
                    lossP = loss_adding(w11*acc_loss[0], w12*Q_loss_plus_batch)
                else:
                    lossP = loss_adding(w11*acc_loss[0], w12*Q_loss_batch)
                
                lossY = loss_adding(w21*acc_loss[1], w22*Q_loss_batch)
            else:
                lossP, lossY = acc_loss

            for opt,(ii,loss),training in zip([optimizerP,optimizerY],enumerate([lossP,lossY]),continue_training):
                train_loss_acc[ii]+=acc_loss[ii].detach().cpu().item()*len(inputs)
                if training:
                    opt.zero_grad()
                    try:
                        loss.backward(retain_graph=True)
                    except RuntimeError as e:
                        print("Raise Error:", e, "when using values:", [i.detach().cpu().numpy() for i in [rpPred, yieldPred, dpReal, rpReal, yieldReal]])
                    opt.step()                

        train_losses.append(train_loss_acc)
        train_objs.append(train_regret)

        if errored_decision!=0:
            all_errored_decisions += errored_decision
            print(epoch, ', Errored decisions:', errored_decision, 'Overall skipped items:', all_errored_decisions)    

        if switch_by_patience:
            if switch_patience!=0 and len(train_losses)>2 and not switched:
                if max(es_counter)>=switch_patience:
                    switched = True
                    # REG = REG_
                    w11,w12,w21,w22 = loss_weights(REG_)
                    if w12+w22!=0:
                        # backward_loss = 'regret'
                        val_method = 'reg'
                        best_loss = np.inf
                        count_q = True
                        best_loss_actual = np.inf
                    continue_training = [True]*len(models)
                    print(f'Loss function changed. Price:{w11}*MSE+{w12}*SPO. Yield:{w21}*MSE+{w22}*SPO.')

                    optimizerP = optim.SGD(models[0].parameters(), lr=lr/lr_decay, weight_decay=1e-5)
                    optimizerY = optim.SGD(models[1].parameters(), lr=lr/lr_decay, weight_decay=1e-5)
                    for (ii,task),model in zip(enumerate(tasks),models):
                        if not (ii==1 and use_forecast):
                            torch.save(model.state_dict(), f'{ckpt_path}/best-model-before-switch_{task}_{model_key}.pth')
                    print('backward loss switched.')
                    if save_after_switch:
                        save_from = epoch+1

        elif switch_patience>0 and not switched:
            if epoch >= switch_patience:
                switched = True
                # REG = REG_
                w11,w12,w21,w22 = loss_weights(REG_)
                if w12+w22!=0:
                    # backward_loss = 'regret'
                    val_method = 'reg'
                    best_loss = np.inf
                    count_q = True
                    best_loss_actual = np.inf
                continue_training = [True]*len(models)
                print(f'Loss function changed. Price:{w11}*MSE+{w12}*SPO. Yield:{w21}*MSE+{w22}*SPO.')

                optimizerP = optim.SGD(models[0].parameters(), lr=lr/lr_decay, weight_decay=1e-5)
                optimizerY = optim.SGD(models[1].parameters(), lr=lr/lr_decay, weight_decay=1e-5)
                for (ii,task),model in zip(enumerate(tasks),models):
                    if not (ii==1 and use_forecast):
                        torch.save(model.state_dict(), f'{ckpt_path}/best-model-before-switch_{task}_{model_key}.pth')
                if save_after_switch:
                    save_from = epoch+1

        if val_loader is not None:
            val_acc, val_regret = test_model(models,tasks,val_loader,count_q,model_key,robust=robust,yield_bias=yield_bias,w_neg=w_neg,w_pos=w_pos,model_type=model_type, path=ckpt_path)
            val_losses.append(val_acc)
            val_objs.append(val_regret)

            best_loss, best_loss_actual, es_counter, continue_training = save_models(models=models,tasks=tasks,val_acc=val_acc,val_regret=val_regret, best_loss=best_loss,best_loss_actual=best_loss_actual,patience=patience,patience_alpha=patience_alpha,epoch=epoch,counter=es_counter, continue_training=continue_training,model_key=model_key,val_method=val_method,path=ckpt_path)

            print(f'Epoch {epoch}/{num_epochs}: training loss={np.array(train_losses[-1]).round(3)}, regret = {train_regret:.3f}; validation loss={np.array(val_losses[-1]).round(3)}, regret = {val_regret:.3f}.')

            if (epoch+1) % save_cycle == 0:
                for obj,name in zip([train_losses, train_objs, val_losses, val_objs],['train-losses', 'train-objs', 'val-losses', 'val-objs']):
                    np.save(f'{name}_{model_key}.npy', np.array(obj))
        else:
            if (epoch+1) % save_cycle == 0:
                print(f'Epoch {epoch}/{num_epochs}: training loss={np.array(train_losses[-1]).round(3)}, regret = {train_regret:.3f}; NO validation.')
                for obj,name in zip([train_losses, train_objs],['train-losses', 'train-objs']):
                    np.save(f'{name}_{model_key}.npy', np.array(obj))

        if all(not x for x in continue_training):
            # print(f'Training stopped at epoch {epoch+1}!')
            break   

        if epoch >= save_from:
            for (ii,task),model in zip(enumerate(tasks),models):
                torch.save(model.state_dict(), f'{ckpt_path}/progress-model_{task}_{model_key}_{epoch}.pth')
    print("=============================================================")

    if test_loader is not None:
        test_loss_acc, test_regret = test_model(models,tasks,test_loader,count_q=True,load_bbm=True, model_key=model_key,robust=robust, model_type=model_type, path=ckpt_path)
        print(f'The training of Model:{model_key} is finished at epoch {epoch+1}. Test accuracy = {np.array(test_loss_acc).round(3)}, overall regret = {test_regret:.3f}.')
    
    if save_prog:
        save_training_summary(train_losses, train_objs, val_losses, val_objs, model_key)

    if val_loader is not None:
        return train_losses, train_objs, val_losses, val_objs
    else:
        return train_losses, train_objs