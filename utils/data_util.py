import pickle
import sklearn.preprocessing
from tslearn.datasets import UCR_UEA_datasets


def read_UCR_UEA(dataset):
    X_train, train_y, X_test, test_y = UCR_UEA_datasets().load_dataset(dataset)
    # X_train = np.nan_to_num(X_train, copy=True, nan=0.0)
    # X_test = np.nan_to_num(X_test, copy=True, nan=0.0)

    train_x = X_train.reshape(-1, X_train.shape[-1], X_train.shape[-2])
    test_x = X_test.reshape(-1, X_train.shape[-1], X_train.shape[-2])
    enc1 = sklearn.preprocessing.OneHotEncoder(sparse_output=False).fit(train_y.reshape(-1, 1))

    train_y = enc1.transform(train_y.reshape(-1, 1))
    test_y = enc1.transform(test_y.reshape(-1, 1))

    return train_x, test_x, train_y, test_y, enc1

