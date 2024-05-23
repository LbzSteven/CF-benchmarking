import json
import os.path
import pickle
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
    if dataset_choice == 'uni':
        univariate_equal_length = [item for item in univariate_equal_length if item not in failed_loading]
        datasets = univariate_equal_length
    elif dataset_choice == 'mul':
        multivariate_equal_length = [item for item in multivariate_equal_length if item not in failed_loading]
        datasets = multivariate_equal_length
    elif dataset_choice == 'all':
        datasets = all_equal_length
    else:
        raise 'wrong set of datasets'

    return datasets


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
