
import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import torch.nn as nn
from model.Model import Traffic_model as Network_Predict
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch, MSE_torch, huber_loss
from lib.Params_pretrain import parse_args
from lib.Params_predictor import get_predictor_params
from lib.data_process import define_dataloder

# *************************************************************************#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_nodes_dict = {'PEMS04': 307, 'PEMS08': 170, 'PEMS07': 883, 'CD_DIDI': 524, 'SZ_DIDI': 627,
                  'METR_LA': 207, 'PEMS_BAY': 325, 'PEMS07M': 228, 'NYC_TAXI': 263, 'CHI_TAXI': 77, 'NYC_BIKE-3': 540,
                  'CAD3': 480, 'CAD4-1': 621, 'CAD4-2': 610, 'CAD4-3': 593, 'CAD4-4': 528, 'CAD5': 211,
                  'CAD7-1': 666, 'CAD7-2': 634, 'CAD7-3': 559, 'CAD8-1': 510, 'CAD8-2': 512, 'CAD12-1': 453, 'CAD12-2': 500,
                  'TrafficZZ': 676, 'TrafficCD': 728, 'TrafficHZ': 672, 'TrafficJN': 576, 'TrafficSH': 896,
                  }

args = parse_args(device)
args.num_nodes_dict = num_nodes_dict
args_predictor = get_predictor_params(args)
attr_list = []
if args.mode !='pretrain':
    for arg in vars(args):
        attr_list.append(arg)
    for attr in attr_list:
        if hasattr(args, attr) and hasattr(args_predictor, attr):
            setattr(args, attr, getattr(args_predictor, attr))
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print('==========')
    for arg in vars(args_predictor):
        print(arg, ':', getattr(args_predictor, arg))
init_seed(args.seed, args.seed_mode)

print('mode: ', args.mode, '  model: ', args.model, '  dataset: ', args.dataset_use, '  load_pretrain_path: ', args.load_pretrain_path, '  save_pretrain_path: ', args.save_pretrain_path)

def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
log_dir = os.path.join(parent_dir,'model_weights', args.model)
Mkdir(log_dir)
args.log_dir = log_dir

#load dataset
train_dataloader, val_dataloader, test_dataloader, scaler_dict = define_dataloder(args=args)

#init model
model = Network_Predict(args, args_predictor)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(args.device)

if args.xavier:
    for p in model.parameters():
        if p.requires_grad==True:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

#init loss function, optimizer
def scaler_mae_loss(mask_value):
    def loss(preds, labels, scaler, mask=None):
        if scaler and args.real_value == False:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if args.mode == 'pretrain' and mask is not None:
            preds = preds * mask
            labels = labels * mask
        mae, mae_loss = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

def scaler_huber_loss(mask_value):
    def loss(preds, labels, scaler, mask=None):
        if scaler and args.real_value == False:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if args.mode == 'pretrain' and mask is not None:
            preds = preds * mask
            labels = labels * mask
        mae = huber_loss(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

if args.loss_func == 'mask_mae':
    loss = scaler_mae_loss(mask_value=args.mape_thresh)
    print('============================scaler_mae_loss')
elif args.loss_func == 'mask_huber':
    if args.mode != 'pretrain':
        loss = scaler_huber_loss(mask_value=args.mape_thresh)
        print('============================scaler_huber_loss')
    else:
        loss = scaler_mae_loss(mask_value=args.mape_thresh)
        print('============================scaler_mae_loss')
    # print(args.model, Mode)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss()
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss()
else:
    raise ValueError


def load_checkpoint(m, ckpt_path):
    if torch.cuda.device_count() > 1:
        m.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    else:
        model_weights = {k.replace('module.', ''): v for k, v in torch.load(ckpt_path, map_location=args.device).items()}
        m.load_state_dict(model_weights)


def get_predictor(m):
    if hasattr(m, 'module'):
        return m.module.predictor
    return m.predictor


def make_optimizer(trainable_only=False):
    params = filter(lambda p: p.requires_grad, model.parameters()) if trainable_only else model.parameters()
    return torch.optim.Adam(params=params, lr=args.lr_init, eps=1.0e-8, weight_decay=0, amsgrad=False)


def make_scheduler(optimizer):
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=lr_decay_steps, gamma=args.lr_decay_rate)
    return None


ckpt_path = log_dir + '/' + args.load_pretrain_path

if args.mode == 'pretrain' or args.mode == 'ori':
    optimizer = make_optimizer(trainable_only=False)
    scheduler = make_scheduler(optimizer)
    trainer = Trainer(model, loss, optimizer, train_dataloader, val_dataloader, test_dataloader, scaler_dict, args, scheduler=scheduler)
    print_model_parameters(model, only_num=False)
    trainer.multi_train()
elif args.mode == 'eval':
    load_checkpoint(model, ckpt_path)
    print("Load saved model")
    for param in model.parameters():
        param.requires_grad = False
    for param in get_predictor(model).linear.parameters():
        param.requires_grad = True
    optimizer = make_optimizer(trainable_only=True)
    scheduler = make_scheduler(optimizer)
    trainer = Trainer(model, loss, optimizer, train_dataloader, val_dataloader, test_dataloader, scaler_dict, args, scheduler=scheduler)
    print_model_parameters(model, only_num=False)
    trainer.multi_train()
elif args.mode == 'lora_eval':
    sys.path.insert(0, parent_dir)
    from repro.lora.apply_lora import apply_lora_to_opencity, freeze_non_lora, trainable_param_count

    load_checkpoint(model, ckpt_path)
    print("Load saved model for LoRA fine-tuning")
    rank = int(getattr(args, 'lora_rank', 8))
    alpha = getattr(args, 'lora_alpha', None)
    if alpha is not None and alpha <= 0:
        alpha = None
    n_layers = apply_lora_to_opencity(get_predictor(model), rank=rank, alpha=alpha)
    freeze_non_lora(model)
    n_train = trainable_param_count(model)
    print(f"LoRA rank={rank} alpha={alpha or 2*rank} replaced={n_layers} trainable_params={n_train}")
    optimizer = make_optimizer(trainable_only=True)
    scheduler = make_scheduler(optimizer)
    trainer = Trainer(model, loss, optimizer, train_dataloader, val_dataloader, test_dataloader, scaler_dict, args, scheduler=scheduler)
    print_model_parameters(model, only_num=False)
    trainer.multi_train()
elif args.mode == 'test':
    optimizer = make_optimizer(trainable_only=False)
    scheduler = make_scheduler(optimizer)
    trainer = Trainer(model, loss, optimizer, train_dataloader, val_dataloader, test_dataloader, scaler_dict, args, scheduler=scheduler)
    print_model_parameters(model, only_num=False)
    print("Load saved model")
    trainer.test(model, trainer.args, scaler_dict, test_dataloader, trainer.logger, path=ckpt_path)
else:
    raise ValueError(f"Unknown mode: {args.mode}")
