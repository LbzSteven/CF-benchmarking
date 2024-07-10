import numpy as np
import pandas as pd
import torch
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance
from utils.data_util import get_CFs
from tsai.models.InceptionTime import InceptionTime
from tsai.models.MLP import MLP
from tsai.models.FCN import FCN
from tsai.models.ResNet import ResNet
def proximity(orig, exp):
    # le=exp.shape[-1]*exp.shape[-2]
    difference = (orig - exp).flatten()
    return np.linalg.norm(difference, 1), np.linalg.norm(difference), np.linalg.norm(difference,
                                                                                     np.inf)  # L1 L2 and L_inf


def mean_absolute_error(orig, exp):
    return np.mean(np.abs(np.array(orig) - np.array(exp)))


def sparsity(orig, exp):
    orig = orig.reshape(-1, 1)
    exp = exp.reshape(-1, 1)
    num_diff_elements = np.sum(orig != exp) / orig.shape[0]
    return num_diff_elements


# def generate_metric_stat(L0, L1, L2, Linf, maes, IM1, AE_loss, generation_times, num_instance, num_valid):
#     return [np.mean(L0), np.std(L0),
#             np.mean(L1), np.std(L1),
#             np.mean(L2), np.std(L2),
#             np.mean(Linf), np.std(Linf),
#             np.mean(maes), np.std(maes),
#             np.mean(IM1), np.std(IM1),
#             np.mean(AE_loss), np.std(AE_loss),
#             np.mean(generation_times), np.std(generation_times),
#             num_valid / num_instance]
def generate_metric_stat(L0, L1, L2, Linf, maes, generation_times, num_instance, num_valid):
    return [np.mean(L0), np.std(L0),
            np.mean(L1), np.std(L1),
            np.mean(L2), np.std(L2),
            np.mean(Linf), np.std(Linf),
            np.mean(maes), np.std(maes),
            # np.mean(IM1), np.std(IM1),
            # np.mean(AE_loss), np.std(AE_loss),
            np.mean(generation_times), np.std(generation_times),
            num_valid / num_instance]

def generate_metric_post_hoc(L0, L1, L2, Linf, maes,  num_instance, num_valid): # notice that this doesn't reevaluate time
    return [np.mean(L0), np.std(L0),
            np.mean(L1), np.std(L1),
            np.mean(L2), np.std(L2),
            np.mean(Linf), np.std(Linf),
            np.mean(maes), np.std(maes),
            num_valid / num_instance]

# def plausibility(AEcf, AEorig, AE, cf, criterion):
#     epsilon = 1e-6
#     _cf = torch.tensor(cf.reshape(1, -1, cf.shape[-1])).float()
#     print(cf.shape, _cf.shape)
#     preds_AEcf = AEcf(_cf)
#     loss_1 = criterion(preds_AEcf, _cf).item()
#     preds_AEorig = AEorig(_cf)
#     loss_2 = criterion(preds_AEorig, _cf).item()
#
#     IM1 = loss_1 / (loss_2 + epsilon)
#
#     preds_AE = AE(_cf)
#     loss_3 = criterion(preds_AEcf, preds_AE).item()
#
#     l1_norm = np.linalg.norm(cf, ord=1)
#     IM2 = loss_3 / (l1_norm + epsilon)
#     return IM1, IM2

def plausibility(cf_label, pred_label, AE_dict, cf, criterion):
    AE_cf = AE_dict[cf_label]
    AE_pred = AE_dict[pred_label]
    AE = AE_dict['AE']
    epsilon = 1e-6
    _cf = torch.tensor(cf.reshape(1, -1, cf.shape[-1])).float()
    # print(cf.shape, _cf.shape)
    preds_AEcf = AE_cf(_cf)
    loss_1 = criterion(preds_AEcf, _cf).item()
    preds_AEorig = AE_pred(_cf)
    loss_2 = criterion(preds_AEorig, _cf).item()

    IM1 = loss_1 / (loss_2 + epsilon)

    preds_AE = AE(_cf)
    loss_3 = criterion(preds_AEcf, preds_AE).item()

    l1_norm = np.linalg.norm(cf, ord=1)
    IM2 = loss_3 / (l1_norm + epsilon)
    return IM1, IM2, loss_1, loss_2, loss_3 #we only need IM1 and loss_3



class OODs:
    def __init__(self, training_set, seeds=range(0, 10)):
        training_set = training_set.reshape(training_set.shape[0], -1)
        self.lof = LocalOutlierFactor(n_neighbors=int(np.sqrt(len(training_set))), novelty=True, metric='euclidean').fit(training_set)
        self.OC_SVM = OneClassSVM(gamma='scale', nu=0.02).fit(training_set)
        self.iforest_list = []
        for seed in seeds:
            iforest = IsolationForest(random_state=seed).fit(training_set)
            self.iforest_list.append(iforest)

    def __call__(self, testing_set):
        novelty_detection_LOF = self.lof.predict(testing_set)
        novelty_detection_OC_SCM = self.OC_SVM.predict(testing_set)
        ood_LOF = np.count_nonzero(novelty_detection_LOF == -1)
        ood_OC_SVM = np.count_nonzero(novelty_detection_OC_SCM == -1)
        ood_LOF_per = ood_LOF / testing_set.shape[0]
        ood_OC_SVM_per = ood_OC_SVM / testing_set.shape[0]
        ood_iforest_list = []
        for i in range(0, 10):
            iforest = self.iforest_list[i]
            novelty_detection_iforest = iforest.predict(testing_set)
            ood_iforest_list.append(np.count_nonzero(novelty_detection_iforest == -1))
        ood_iforest_per = np.mean(np.array(ood_iforest_list)) / testing_set.shape[0]
        ood_iforest_per_std = np.round(np.std(np.array(ood_iforest_list)) / testing_set.shape[0], 3)
        return ood_LOF_per, ood_OC_SVM_per, ood_iforest_per, ood_iforest_per_std

    def get_OC_SVM(self):
        return self.OC_SVM

    def get_lof(self):
        return self.lof

    def get_iforest_list(self):
        return self.iforest_list


def post_evaluate_CFs_based_on_datasets(exp_results, test_x, valid):
    L0 = []
    L1 = []
    L2 = []
    Linf = []
    maes = []
    num_instance = test_x.shape[0]
    in_channels = test_x.shape[-2]
    conresponding_test_x = test_x[valid].copy()
    num_valid = len(valid)
    for i in range(num_valid):
        orig = conresponding_test_x[i].reshape(in_channels, -1)
        CF = exp_results[i].reshape(in_channels, -1)
        L0.append(sparsity(orig, CF))
        L1_i, L2_i, Linf_i = proximity(orig, CF)
        L1.append(L1_i)
        L2.append(L2_i)
        Linf.append(Linf_i)
        maes.append(mean_absolute_error(orig, CF))

    return generate_metric_post_hoc(L0, L1, L2, Linf, maes, num_instance, num_valid)

def compute_metric_given_path(path):
    valid,tx,ty,pred,cf,cf_pred = get_CFs(path)
    if 'wCF' in path or 'TSEvo' in path:
        num_instance = len(tx)
        if num_instance>20:
            np.random.seed(42)
            random_selection = np.random.choice(num_instance, size=20, replace=False)
        else:
            random_selection = np.arange(num_instance)
        tx_selected = tx[random_selection].copy()
    else:
        tx_selected = tx.copy()
    return post_evaluate_CFs_based_on_datasets(cf,tx_selected,valid)


# Plausibility
def get_hidden_layers(model, hook_block, data, device='cpu'):
    latent_representation = {}
    if hook_block is None:
        if isinstance(model,FCN) or isinstance(model,InceptionTime):
            hook_block = model.gap.gap
        elif isinstance(model,ResNet):
            hook_block = model.gap
        elif isinstance(model,MLP):
            hook_block = model.mlp[2][2]
        else:
            raise "Unspecified model"
    def forward_hook(name):
        def hook(model, input, output):
            latent_representation[name] = output.detach().cpu()

        return hook

    features = data.shape[-2]
    length = data.shape[-1]
    for i in range(len(data)):
        handle = hook_block.register_forward_hook(forward_hook(i))
        input_data = torch.from_numpy(data[i].reshape(1, features, length)).float().to(device)
        output = model(input_data)
        handle.remove()

    return np.vstack(list(latent_representation.values()))


def get_distance_latent(CF_latent:np.ndarray, cf_pred, train_x_latent:np.ndarray,train_pred, metric=distance.euclidean):
    def get_train_x_latent_classwise(train_x_latent: np.ndarray, train_pred):
        train_x_latent_classwise = {}
        # devide different train activations for different groups
        for i in np.unique(train_pred):
            keys = np.where(train_pred == i)[0].tolist()
            filtered_dict = train_x_latent[keys].copy()
            train_x_latent_classwise[i] = filtered_dict
        return train_x_latent_classwise
    d2_list = []

    train_x_latent_classwise= get_train_x_latent_classwise(train_x_latent,train_pred)

    for i in range(len(cf_pred)):
        NG_prediction = cf_pred[i]
        current_CF_latent = CF_latent[i].flatten()

        target_class_latent:np.ndarray = train_x_latent_classwise[NG_prediction]
        target_class_num = target_class_latent.shape[0]
        d2 = np.min(
            [metric(current_CF_latent, target_class_latent[j].flatten()) for j in range(target_class_num)])
        # d2  is the distance from CF to the target class

        d2_list.append(d2)
    return d2_list