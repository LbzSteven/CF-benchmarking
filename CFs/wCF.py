from typing import Tuple
import numpy as np
from TSInterpret.InterpretabilityModels.counterfactual.CF import CF
# from TSInterpret.Models.PyTorchModel import PyTorchModel
from .PyTorchModel import PyTorchModel
from TSInterpret.Models.SklearnModel import SklearnModel
from TSInterpret.Models.TensorflowModel import TensorFlowModel
from scipy.stats import stats
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist

def dist_mad(orig, cf, mad):
    manhat = np.abs(orig - cf)
    return np.sum((manhat / mad).flatten())


def loss_function_mad(x_dash, predict, TS_length, feature_num, target, Lambda, orig, mad):
    _x_dash = x_dash.reshape(1, feature_num, TS_length)
    # print(_x_dash.shape)
    L = Lambda * (predict(_x_dash)[0][target] - 1) ** 2 + dist_mad(orig, _x_dash, mad)
    return L


class wCF(CF):
    """
    reference: https://arxiv.org/abs/1711.00399
    https://github.com/e-delaney/Instance-Based_CFE_TSC/tree/main/W-CF
    """

    def __init__(
            self,
            model,
            X_train,
            backend,
            mode,
            max_iter=500,
            lambda_init=10,
            pred_threshold=0.5,
            device='cuda:0'
    ) -> None:
        super().__init__(model, mode)
        shape = X_train.shape
        print(shape)
        if mode == "time":
            # Parse test data into (1, feat, time):
            change = True
            X_train = np.swapaxes(X_train, 2, 1)
        elif mode == "feat":
            change = False
        self.ts_length = shape[-1]
        self.num_feature = shape[-2]
        self.device = device
        self.model.to(device)
        self.mad = stats.median_abs_deviation(X_train)
        if backend == "PYT":
            self.predict = PyTorchModel(model, change, device).predict
        elif backend == "TF":
            self.predict = TensorFlowModel(model, change).predict
        elif backend == "SK":
            self.predict = SklearnModel(model, change).predict
        self.max_iter = max_iter
        self.lambda_init = lambda_init
        self.target = None
        self.mad = stats.median_abs_deviation(X_train, axis=0)
        self.pred_threshold = pred_threshold
    def explain(self, x: np.ndarray, orig_class: int = None, target: int = None) -> Tuple[np.ndarray, int]:
        print(x.shape)
        if self.mode != "feat":
            x = np.swapaxes(x, -1, -2)
        self.orig = x.copy()
        self.cf = x.copy()
        print(x.shape)
        if target is None:
            self.target = np.argsort((self.predict(x)))[0][-2:-1][0]
        print(self.predict(x))
        print(self.target)
        prob_target = self.predict(self.cf)[0][self.target]
        i = 0
        while prob_target < self.pred_threshold:
            Lambda = self.lambda_init * (1 + 0.5) ** i
            print(f'{i} round, prob:{prob_target},lamda:{Lambda}')
            res = minimize(loss_function_mad, self.cf.flatten(), args=(self.predict, self.ts_length, self.num_feature, self.target, Lambda, self.orig, self.mad), \
                           method='nelder-mead', options={'maxiter': 10, 'xatol': 50, 'adaptive': True})
            self.cf = res.x.reshape(1, self.num_feature, self.ts_length)
            prob_target = self.predict(self.cf)[0][self.target]
            i += 1
            if i == self.max_iter:
                return None, None
        pred_cf = np.argmax(self.predict(self.cf)[0])
        return self.cf, pred_cf

