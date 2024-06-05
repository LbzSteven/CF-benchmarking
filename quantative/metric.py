import numpy as np
import pandas as pd
import torch
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


def proximity(orig, exp):
    # le=exp.shape[-1]*exp.shape[-2]

    return np.linalg.norm(orig - exp, 1), np.linalg.norm(orig - exp), np.linalg.norm(orig - exp,
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
