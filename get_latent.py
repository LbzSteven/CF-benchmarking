import os
import pickle
import json
import numpy as np
import torch
from tqdm import trange
from tslearn.datasets import UCR_UEA_datasets

from utils.data_util import read_UCR_UEA, get_CFs, get_UCR_UEA_sets
from utils.model_util import model_init
from utils.train_util import generate_loader, get_all_preds
from quantative.metric import get_hidden_layers, get_distance_latent,compute_plausibility_dataset


def get_latent_model(dataset, model_name, device='cuda:0',save=True):
    model_dataset_path = f'../models/{model_name}/{dataset}'
    train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA(dataset=dataset, UCR_UEA_dataloader=UCR_UEA_datasets())

    _, train_loader_no_shuffle = generate_loader(train_x, train_x, train_y, train_y)

    in_channels = train_x.shape[-2]
    input_size = train_x.shape[-1]
    n_pred_classes = train_y.shape[1]
    model = model_init(model_name, in_channels=in_channels, n_pred_classes=n_pred_classes, seq_len=input_size)
    state_dict = torch.load(f'{model_dataset_path}/weight.pt')
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    train_preds, _ = get_all_preds(model, train_loader_no_shuffle, device=device)
    train_preds = np.array(train_preds)

    # put them in get latent
    train_latent:np.ndarray = get_hidden_layers(model=model,hook_block=None,data=train_x,device=device)
    test_latent:np.ndarray  = get_hidden_layers(model=model,hook_block=None,data=test_x,device=device)
    # Save both train and test in model_dataset_path
    print(train_latent.shape)
    print(test_latent.shape)
    if save:

        np.save(f'{model_dataset_path}/train_latent.npy', train_latent)
        np.save(f'{model_dataset_path}/test_latent.npy', test_latent)
        np.save(f'{model_dataset_path}/train_preds.npy',train_preds)
    return  train_latent,test_latent,train_preds


def get_latent_CF(dataset, model_name, CF_method='NG', device='cuda:0',save=True):
    model_dataset_path = f'../models/{model_name}/{dataset}'
    CF_path = f'../CF_result/{CF_method}/{model_name}/{dataset}/'
    train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA(dataset=dataset, UCR_UEA_dataloader=UCR_UEA_datasets())
    if os.path.exists(CF_path):
        print(f'{CF_path} exist')
    else:
        raise f'{CF_method} not evaluated on {dataset} {model_name}'

    _, _, _, _, exp_results, cf_preds = get_CFs(CF_path)
    in_channels = train_x.shape[-2]
    input_size = train_x.shape[-1]
    n_pred_classes = train_y.shape[1]
    model = model_init(model_name, in_channels=in_channels, n_pred_classes=n_pred_classes, seq_len=input_size)
    state_dict = torch.load(f'{model_dataset_path}/weight.pt')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    CF_latent: np.ndarray  = get_hidden_layers(model=model, hook_block=None, data=exp_results, device=device)
    print(CF_latent.shape)
    if save:
        np.save(f'{CF_path}/CF_latent.npy',CF_latent)
    return CF_latent,cf_preds

def generate_CFs_latent(CF_name, model_name, dataset_choice, device: str = 'cuda:0', start_per: float = 0.0, end_per: float = 1.0,save=True):
    datasets = get_UCR_UEA_sets(dataset_choice)
    UCR_UEA_dataloader = UCR_UEA_datasets()
    if model_name == 'all':
        models = ['ResNet', 'FCN', 'InceptionTime', 'MLP']
    else:
        models = [model_name]

    total_length = len(datasets)
    start = int(start_per * total_length)
    end = int(end_per * total_length)
    print(f'total dataset length:{total_length}')
    print(f'starting:{start},ending:{end}')
    for model_name in models:
        pbar = trange(end - start, desc='Dataset', unit='epoch', initial=0, disable=False)
        # method_record = get_result_JSON(model_name)
        # recorded_dataset = method_record.keys()
        for i in range(start, end):
            dataset = datasets[i]
            if dataset == 'Lightning2' and CF_name == 'SETS':
                continue

            get_latent_CF(dataset, model_name, CF_method=CF_name, device=device, save=save)

            pbar.update(1)

def generate_models_latent(model_name, dataset_choice, device: str = 'cuda:0', start_per: float = 0.0, end_per: float = 1.0, save=True):
    datasets = get_UCR_UEA_sets(dataset_choice)

    if model_name == 'all':
        models = ['ResNet', 'FCN', 'InceptionTime', 'MLP']
    else:
        models = [model_name]

    total_length = len(datasets)
    start = int(start_per * total_length)
    end = int(end_per * total_length)
    print(f'total dataset length:{total_length}')
    print(f'starting:{start},ending:{end}')
    for model_name in models:
        pbar = trange(end - start, desc='Dataset', unit='epoch', initial=0, disable=False)
        # method_record = get_result_JSON(model_name)
        # recorded_dataset = method_record.keys()
        for i in range(start, end):
            dataset = datasets[i]
            get_latent_model(dataset, model_name, device=device,save=save)

            pbar.update(1)

if __name__ == '__main__':
    # 0613 sanity check
    # get_latent_model('GunPoint', 'FCN', device='cuda:1')
    # model_name = 'FCN'
    # get_latent_CF('GunPoint', 'MLP', CF_method='wCF', device='cuda:1', save=True)
    # train_latent,test_latent,train_preds = get_latent_model('GunPoint', 'MLP', device='cuda:1',save=False)
    # CF_latent,cf_preds = get_latent_CF('GunPoint', 'MLP', CF_method='wCF', device='cuda:1',save=False)
    # d2 = get_distance_latent(CF_latent=CF_latent,cf_pred=cf_preds,train_x_latent=train_latent,train_pred=train_preds)
    # print(d2)

    dataset_choice = 'selected_uni'
    # for model_name in ['InceptionTime', 'MLP','FCN','ResNet']:
    #     generate_models_latent(model_name, dataset_choice,device='cuda:0', start_per= 0.0, end_per = 1.0, save = True)

    for model_name in ['InceptionTime', 'MLP','FCN']:
        for CF_name in ['NUN_CF','NG','wCF','TSEvo','SETS']:
            if model_name == 'MLP' and CF_name=='NG':
                continue
            generate_CFs_latent(CF_name,model_name, dataset_choice,device='cuda:0', start_per= 0.0, end_per = 1.0, save = True)