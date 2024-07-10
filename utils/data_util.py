import json
import os.path
import pickle

import numpy as np
import pandas as pd
import sklearn.preprocessing
from tslearn.datasets import UCR_UEA_datasets


def read_UCR_UEA(dataset, UCR_UEA_dataloader):
    if UCR_UEA_dataloader is None:
        UCR_UEA_dataloader = UCR_UEA_datasets()
    X_train, train_y, X_test, test_y = UCR_UEA_dataloader.load_dataset(dataset)
    # X_train = np.nan_to_num(X_train, copy=True, nan=0.0)
    # X_test = np.nan_to_num(X_test, copy=True, nan=0.0)
    if X_train is None:
        print(f"{dataset} could not load correctly")
        return None, None, None, None, None
    train_x = X_train.reshape(-1, X_train.shape[-1], X_train.shape[-2])
    test_x = X_test.reshape(-1, X_train.shape[-1], X_train.shape[-2])
    enc1 = sklearn.preprocessing.OneHotEncoder(sparse_output=False).fit(train_y.reshape(-1, 1))

    train_y = enc1.transform(train_y.reshape(-1, 1))
    test_y = enc1.transform(test_y.reshape(-1, 1))

    return train_x, test_x, train_y, test_y, enc1


def get_UCR_UEA_sets(dataset_choice: str) -> list:
    with open('./JSON/UCR_UEA.json', 'rb') as file:
        UCR_UEA = json.load(file)
    # uni = UCR_UEA_dataloader.list_univariate_datasets()
    # mul = UCR_UEA_dataloader.list_multivariate_datasets()
    univariate = UCR_UEA['univariate']
    univariate2015 = UCR_UEA['univariate2015']
    univariate_equal_length = UCR_UEA['univariate_equal_length']
    multivariate = UCR_UEA['multivariate']
    multivariate_equal_length = UCR_UEA['multivariate_equal_length']
    univariate_equal_length_2015 = UCR_UEA['univariate_equal_length_2015']
    univariate_equal_length_new = UCR_UEA['univariate_equal_length_new']
    failed_loading = UCR_UEA['failed_loading']

    all_equal_length: list = univariate_equal_length + multivariate_equal_length
    all_equal_length = [item for item in all_equal_length if item not in failed_loading]

    selected_uni = UCR_UEA['selected_uni']
    selected_uni = [item for item in selected_uni if item in all_equal_length]
    selected_uni_ordered = ['Computers', 'ElectricDevices', 'ECG200', 'ECG5000', 'NonInvasiveFetalECGThorax1',
                            'DistalPhalanxOutlineCorrect', 'HandOutlines', 'ShapesAll', 'Yoga', 'GunPoint',
                            'UWaveGestureLibraryAll', 'PowerCons', 'Earthquakes', 'FordA', 'Wafer', 'CBF',
                            'TwoPatterns', 'Beef', 'Strawberry', 'Chinatown']
    selected_mul_ordered = ['Heartbeat', 'StandWalkJump', 'SelfRegulationSCP1', 'Cricket',
                             'BasicMotions', 'Epilepsy', 'NATOPS', 'RacketSports', 'EigenWorms','Libras','UWaveGestureLibrary'
                            'Phoneme']
    selected_mul = UCR_UEA['selected_mul']
    selected_mul = [item for item in selected_mul if item in all_equal_length]
    selected_all = selected_uni + selected_mul
    if dataset_choice == 'uni':
        univariate_equal_length = [item for item in univariate_equal_length if item not in failed_loading]
        datasets = univariate_equal_length
    elif dataset_choice == 'mul':
        multivariate_equal_length = [item for item in multivariate_equal_length if item not in failed_loading]
        datasets = multivariate_equal_length
    elif dataset_choice == 'all':
        datasets = all_equal_length
    elif dataset_choice == 'selected_uni':
        datasets = selected_uni_ordered
    elif dataset_choice == 'selected_mul':
        datasets = selected_mul_ordered
    elif dataset_choice == 'selected_all':
        datasets = selected_all
    elif dataset_choice == 'multiclass_uni':
        datasets =['ElectricDevices', 'ECG5000', 'NonInvasiveFetalECGThorax1', 'ShapesAll', 'UWaveGestureLibraryAll', 'CBF',
         'TwoPatterns', 'Beef']
    elif (dataset_choice in selected_uni) or (dataset_choice in selected_mul):
        datasets= [dataset_choice]
    else:
        raise 'wrong set of datasets'

    return datasets

def get_reference_UCR():
    full_path = f'./JSON/reference_UCR.json'
    with open(full_path, 'r') as file:
        record_dict = json.load(file)
    return record_dict
def get_result_JSON(method_name) -> dict:
    full_path = f'./JSON/{method_name}.json'
    if not os.path.exists(full_path):
        record_dict = {}
        with open(full_path, 'w') as file:
            json.dump(record_dict, file, indent=4)
    else:
        with open(full_path, 'r') as file:
            record_dict = json.load(file)
    return record_dict


def save_result_JSON(method_name, record_dict):
    full_path = f'./JSON/{method_name}.json'

    with open(full_path, 'w') as file:
        json.dump(record_dict, file, indent=4)

def get_instance_result_JSON(CF_path):
    full_path_result = os.path.join(CF_path,"results.json")
    full_path_plau = os.path.join(CF_path,"plausibility.json")
    with open(full_path_result, 'r') as file:
            result_dict:dict = json.load(file)
    with open(full_path_plau, 'r') as file:
            plau_dict:dict = json.load(file)
    return result_dict, plau_dict

def sample_indices_by_label(labels, selected_size=250):
    """
    Sample indices from each label group in the dataset.

    Parameters:
    - labels (list or numpy array): Array of labels.
    - n_instances (int): Number of instances to sample from each label.

    Returns:
    - selected_indices (list): List of indices of selected instances in the original dataset.
    """
    # Convert labels to DataFrame


    selected_indices = []
    df = pd.DataFrame({
        'labels': labels
    })
    num_instance = len(df)
    selected_size = min(num_instance, selected_size)
    frac = selected_size / len(df)
    for label, group in df.groupby('labels'):
        sampled_indices = group.sample(frac=frac, replace=False, random_state=42).index.tolist()

        selected_indices.extend(sampled_indices)
    selected_indices.sort()
    return selected_indices


def get_CFs(CF_path):
    """

    Parameters
    ----------
    CF_path:  the CF path like './CF_result/NG/FCN/GunPoint/'

    Returns:  valid, test_x, test_y, y_pred, exp_results, pred_CFs,
    -------

    """
    exp_results = np.load(f'{CF_path}/CF.npy')
    test_x = np.load(f'{CF_path}/test_x.npy')
    test_y = np.load(f'{CF_path}/test_y.npy')
    valid = np.load(f'{CF_path}/valid.npy')
    pred_CFs = np.load(f'{CF_path}/pred_CFs.npy')
    if os.path.exists(f'{CF_path}/test_pred.npy'):
        y_pred = np.load(f'{CF_path}/test_pred.npy')
    else:
        y_pred = np.load(f'{CF_path}/y_pred.npy')
    return valid, test_x, test_y, y_pred, exp_results, pred_CFs,

def get_valid_CF_given_path(path):
    valid,tx,ty,pred,cf,cf_pred = get_CFs(path)

    num_instance = len(tx)
    if 'wCF' in path or 'TSEvo' in path:

        np.random.seed(42)
        random_selection = np.load(f'{path}/iterator.npy')

        # tx_selected = tx[random_selection]
        # ty_selected = ty[random_selection]
        # pred_selected = pred[random_selection]
        num_instance = len(random_selection)
    else:
        random_selection = range(num_instance)

    tx_valid = tx[valid]
    ty_valid = ty[valid]
    pred_valid = pred[valid]
    return valid,tx_valid,ty_valid,pred_valid,cf,cf_pred, random_selection, num_instance

def get_CF_dataset_metric(df, CF_names=None, datasets=None, metrics=None,ordered=None):
    df =df.copy()
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = df[numeric_columns].round(3)
    if CF_names is None:
        CF_names = df['CF_name'].unique().tolist()
    if datasets is None:
        datasets = df['dataset_name'].unique().tolist()
    if metrics is None:
        metrics = ['L0', 'L1', 'L2',  'Linf',  'maes',  'gtime', 'valid','all','classwise']
    elif metrics == 'sparsity':
        metrics = ['L0']
    elif metrics == 'proximity':
        metrics = ['L1','L2', 'Linf', 'maes']
    elif metrics == 'plausibility':
        metrics = ['all','classwise']

    if not isinstance(CF_names, list):
        CF_names = [CF_names]
    if not isinstance(datasets,list):
        datasets = [datasets]
    if not isinstance(metrics,list):
        metrics = [metrics]
    for metric in metrics:
        if metric+'_std' in df.columns:
            df[metric] = df.apply(lambda row: f"{row[metric]}({row[metric+'_std']})", axis=1)
    # print(metrics)
    columns = ['CF_name', 'dataset_name'] + metrics
    df_output = df[columns].copy()
    df_output = df_output[(df_output['CF_name'].isin(CF_names)) & (df_output['dataset_name'].isin(datasets))].copy()
    CF_order = ['NUN_CF','wCF','NG','COMTE','TSEvo','SETS']
    df_output['CF_name'] = pd.Categorical(df_output['CF_name'], categories=CF_order, ordered=True)
    df_output = df_output.sort_values('CF_name').reset_index(drop=True)



    if ordered is not None:
        df_output['dataset_name'] = pd.Categorical(df_output['dataset_name'], categories=ordered, ordered=True)
        df_output = df_output.sort_values('dataset_name').reset_index(drop=True)
    return df_output
