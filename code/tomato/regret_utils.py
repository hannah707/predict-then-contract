#%%
import numpy as np

import torch
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.dataset import random_split

import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# import tqdm
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import diffcp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#tomato
ac = -0.11304933813851573
bc = 2.011524763760528 + 0.075
al = -0.15236688050442163
bl = 1.9911915504841364 - 1.6
ap = -0.1
bp = 0.002
p0 = 1.570*3 # contract breach when yl_para<decision_var1
p1 = 0.01 # sell over-contracting (decision_var - contract_capacity) to the market
p2 = 1.570 # waste = average rtPrice
yl_rmse = 0.12833993531028856
pr_rmse = 0.45754691131428116

# #wind
# ac = -0.2693033920958892
# bc = 0.6739350804897631 + 0.1
# al = -0.45389688835410713
# bl = 0.5529071058454333 + 0.06
# ap = -0.0005
# bp = 0.005
# p0 = 0.578 # contract breach when yl_para<decision_var1
# p1 = 0.001 # sell over-contracting (decision_var - contract_capacity) to the market
# p2 = 0.578*3 # waste

# yl_rmse = 0.2675182709796017
# pr_rmse = 0.03826457329030865
# # pc_rmse = 0.1229673618349879

decision_var = cp.Variable(1)

pc_para = cp.Parameter(1)
pr_para = cp.Parameter(1)
yl_para = cp.Parameter(1, nonneg=True)

pc2_para = cp.Parameter(1, nonneg=True) # = pc_para@pc_para
pr2_para = cp.Parameter(1, nonneg=True) # = pr_para@pr_para

pcy_para = cp.Parameter(1, nonneg=True) # = pc_para@yl_para
pry_para = cp.Parameter(1, nonneg=True) # = pr_para@yl_para

constraints = [decision_var>=0, decision_var<=yl_para, decision_var<=ac*pc_para+bc]
objective = cp.Maximize( \
            cp.minimum(pc_para@decision_var, pcy_para, (ac*pc2_para+bc*pc_para)) \
            + cp.minimum((pry_para-pr_para@decision_var), (al*pr2_para+bl*pr_para)) \
            - cp.pos(ap*yl_para+bp) \
            - p0*cp.pos(decision_var - yl_para) \
            - p1*cp.pos(decision_var - (ac*pc_para+bc)) \
            - p2*cp.pos(yl_para - cp.minimum((yl_para-decision_var),(al*pr_para+bl)) - cp.minimum(decision_var,yl_para,ac*pc_para+bc)) \
            # + pos_const \
            )

problem = cp.Problem(objective, constraints)
assert problem.is_dpp(), 'The problem is not DPP-compliant.'

clayer = CvxpyLayer(problem, parameters=[pc_para, pr_para, yl_para, pc2_para, pr2_para, pcy_para, pry_para], variables=[decision_var])

#%%
def calculate_profit_tensor(decision,pc,_pr,_yl):
    decision,pc,_pr,_yl = [torch.tensor(item).to(device) if not isinstance(item, torch.Tensor) else item for item in [decision,pc,_pr,_yl]]
    dl = al*_pr+bl
    dc = ac*pc+bc

    sell_by_contract = torch.clamp(torch.min(torch.min(dc,_yl),decision),min=0)
    sell_at_market = torch.clamp(torch.min(_yl-sell_by_contract, dl),min=0)
    over_production = torch.clamp(_yl-sell_by_contract-sell_at_market,min=0)

    c1 = ap*_yl+bp
    c2 = p0*torch.clamp(decision-_yl,min=0) + \
        p1*torch.clamp(decision-dc,min=0) + \
        p2*over_production   

    obj = pc*sell_by_contract + _pr*sell_at_market - c1 - c2

    return obj

def calculate_profit(decision,pc,_pr,_yl):
    decision,pc,_pr,_yl = [item.item() if isinstance(item, torch.Tensor) else item for item in [decision,pc,_pr,_yl]]
    dl = al*_pr+bl
    dc = ac*pc+bc
        
    sell_by_contract = np.max(np.min([dc,_yl,decision]),0)
    sell_at_market = np.max(np.min([_yl-sell_by_contract, dl]),0)
    # sell_at_market_ = np.min([_yl-decision, dl])
    over_production = np.max([_yl-sell_by_contract-sell_at_market, 0])
    
    c1 = ap*(_yl)+bp
    c2 = p0*np.max([0, decision-_yl]) + p1*np.max([0, decision-dc]) + p2*over_production

    profit = pc*sell_by_contract + _pr*sell_at_market - c1 - c2

    return profit


def Q_loss(realtime_price, yield_pred, pcReal, prReal, yieldReal, pc_rmse=0, pr_rmse=0, yl_rmse=yl_rmse, robust=False, spec=False):
    Q_loss_batch = None
    errored_decision = 0
    Q_losses = []

    for _pr,_yl,pc,pr,yl in zip(realtime_price, yield_pred, pcReal, prReal, yieldReal):
        se=False # True if there is a solver error
        if robust:
            _yl = torch.clamp(_yl-robust*yl_rmse,min=0)
            # _pr = _pr-2*pr_rmse

        x_opt, = clayer(pc, pr, yl, pc*pc, pr*pr, pc*yl, pr*yl)
        profit_x_opt = calculate_profit_tensor(x_opt, pc, pr, yl)

        try:
            x_sol, = clayer(pc, _pr, _yl, pc*pc, _pr*_pr, pc*_yl, _pr*_yl)
            profit_x_sol = calculate_profit_tensor(x_sol, pc, pr, yl)
        except diffcp.cone_program.SolverError as e:
            se=True
            print("Raise Error:", e, "when using values:", [i.detach().cpu().numpy() for i in [pc, _pr, _yl, pc, pr, yl]])
            x_sol = torch.tensor([0.0]).to(torch.float32).to(device)
            profit_x_sol = calculate_profit_tensor(x_sol, pc, pr, yl)
        except AssertionError as e:
            se=True
            print("Raise Error:", e, "when using values:", [i.detach().cpu().numpy() for i in [pc, _pr, _yl, pc, pr, yl]])
            x_sol = torch.tensor([0.0]).to(torch.float32).to(device)
            profit_x_sol = calculate_profit_tensor(x_sol, pc, pr, yl)

        Q_loss = profit_x_opt - profit_x_sol
        Q_losses.append(Q_loss.item())

        if se:
            errored_decision += 1
            print(f'SolverError. So we skipped {errored_decision} decision. So far we have skipped') 
        
        if Q_loss_batch is None:
            Q_loss_batch = Q_loss
        else:
            Q_loss_batch = torch.add(Q_loss_batch, Q_loss)

    if spec:
        return Q_loss_batch, Q_losses, errored_decision
    else:
        return Q_loss_batch, errored_decision

def Q_loss_spoplus(realtime_price, yield_pred, pcReal, prReal, yieldReal, pc_rmse=0, pr_rmse=0, yl_rmse=yl_rmse, robust=False, spec=False):
    Q_loss_batch = None
    errored_decision = 0
    Q_losses = []

    for _pr,_yl,pc,pr,yl in zip(realtime_price, yield_pred, pcReal, prReal, yieldReal):
        se=False # True if there is a solver error
        if robust:
            _yl = torch.clamp(_yl-robust*yl_rmse,min=0)
            # pc = pc-2*pc_rmse
            # _pr = _pr-2*pr_rmse

        x_opt, = clayer(pc, pr, yl, pc*pc, pr*pr, pc*yl, pr*yl)
        profit_x_opt = calculate_profit_tensor(x_opt, pc, pr, yl)
        profit_2chat = calculate_profit_tensor(x_opt, pc, 2*_pr, yl)

        try:
            pr_support = torch.clamp(pr-2*_pr,min=0)
            x_support, = clayer(pc, pr-2*_pr, _yl, pc*pc, pr_support*pr, pc*_yl, pr_support*_yl)
            profit_x_support = calculate_profit_tensor(x_support, pc, pr-2*_pr, _yl)
        except diffcp.cone_program.SolverError as e:
            se=True
            print("Raise Error:", e, "when using values:", [i.detach().cpu().numpy() for i in [pc, _pr, _yl, pc, pr, yl]])
            x_sol = torch.tensor([0.0]).to(torch.float32).to(device)
            profit_x_support = calculate_profit_tensor(x_sol, pc, pr, yl)
        except AssertionError as e:
            se=True
            print("Raise Error:", e, "when using values:", [i.detach().cpu().numpy() for i in [pc, _pr, _yl, pc, pr, yl]])
            x_sol = torch.tensor([0.0]).to(torch.float32).to(device)
            profit_x_support = calculate_profit_tensor(x_sol, pc, pr, yl)

        Q_loss = profit_x_opt - profit_x_support - profit_2chat
        Q_losses.append(Q_loss.item())

        if se:
            errored_decision += 1
            print(f'SolverError. So we skipped {errored_decision} decision. So far we have skipped') 
        
        if Q_loss_batch is None:
            Q_loss_batch = Q_loss
        else:
            Q_loss_batch += Q_loss

    if spec:
        return Q_loss_batch, Q_losses, errored_decision
    else:
        return Q_loss_batch, errored_decision