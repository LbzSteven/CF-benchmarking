"""Implementation after Delaney et al . https://github.com/e-delaney/Instance-Based_CFE_TSC"""
import warnings
from typing import Tuple
from collections import OrderedDict
from typing import Dict, Callable

import torch

import numpy as np
from tsai.models.InceptionTime import InceptionTime
from tsai.models.MLP import MLP
from tsai.models.FCN import FCN
from tsai.models.ResNet import ResNet
from torchcam.methods import CAM
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.neighbors import KNeighborsTimeSeries

from TSInterpret.InterpretabilityModels.counterfactual.CF import CF
from TSInterpret.InterpretabilityModels.GradCam.GradCam_1D import GradCam1D
# from TSInterpret.Models.PyTorchModel import PyTorchModel
from .PyTorchModel import PyTorchModel
from TSInterpret.Models.TensorflowModel import TensorFlowModel

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


class NGCF(CF):

    def __init__(
            self,
            model,
            data,
            backend="PYT",
            mode="feat",
            method="NUN_CF",
            distance_measure="euclidean",
            n_neighbors=1,
            max_iter=500,
            device='cuda:0'
    ) -> None:

        super().__init__(model, mode)

        self.backend = backend
        test_x, y_pred = data
        test_x = np.array(test_x)  # , dtype=np.float32) # Ziwen: this should be call as train_x
        shape = (test_x.shape[-2], test_x.shape[-1])
        if mode == "time":
            # Parse test data into (1, feat, time):
            self.ts_length = test_x.shape[-2]
            test_x = test_x.reshape(test_x.shape[0], test_x.shape[2], test_x.shape[1])
        elif mode == "feat":
            self.ts_length = test_x.shape[-1]
        self.num_feature = test_x.shape[-2]
        self.model.to(device)

        if backend == "PYT":
            self.remove_all_hooks(self.model)
            # try:
            if isinstance(model, MLP):
                self.cam_extractor = None
            else:
                self.cam_extractor = CAM(self.model, input_shape=shape)
            # except:
            #    print("GradCam Hook already registered")
            change = False
            if self.mode == "time":
                change = True
            self.predict = PyTorchModel(self.model, change, device).predict
            # y_pred = np.argmax(self.predict(test_x), axis=1)

        elif backend == "TF":
            self.cam_extractor = GradCam1D()  # VanillaGradients()#GradCam1D()
            # y_pred = np.argmax(
            #     self.model.predict(test_x.reshape(-1, self.ts_length, 1)), axis=1
            # )
            self.predict = TensorFlowModel(self.model, change=True).predict
        else:
            print("Only Compatible with Tensorflow (TF) or Pytorch (PYT)!")

        self.data = (test_x, y_pred)
        self.method = method
        self.distance_measure = distance_measure
        self.max_iter = max_iter
        self.n_neighbors = n_neighbors
        # Manipulate reference set replace original y with predicted y

    def remove_all_hooks(self, model: torch.nn.Module) -> None:
        # TODO Move THIS TO TSINTERPRET !
        if hasattr(model, "_forward_hooks"):
            if model._forward_hooks != OrderedDict():
                model._forward_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(model, "_forward_pre_hooks"):
                model._forward_pre_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(model, "_backward_hooks"):
                model._backward_hooks: Dict[int, Callable] = OrderedDict()
        for name, child in model._modules.items():
            if child is not None:
                if hasattr(child, "_forward_hooks"):
                    child._forward_hooks: Dict[int, Callable] = OrderedDict()
                elif hasattr(child, "_forward_pre_hooks"):
                    child._forward_pre_hooks: Dict[int, Callable] = OrderedDict()
                elif hasattr(child, "_backward_hooks"):
                    child._backward_hooks: Dict[int, Callable] = OrderedDict()
                self.remove_all_hooks(child)

    def _nearest_unlike_retrieval(self, query, predicted_label, target, distance, n_neighbors):
        """
        This gets the nearest unlike neighbors.
        Arguments:
            query (np.array): The instance to explain.
            predicted_label (np.array): Label of instance.
            reference_set (np.array): Set of addtional labeled data (could be training or test set)
            distance (str):
            n_neighbors (int):number nearest neighbors to return
        Returns:
            [np.array]: Returns K_Nearest_Neighbors of input query with different classification label.

        """
        if not isinstance(predicted_label, (int, np.int64, np.int32, np.int16)):
            if len(predicted_label) > 1:
                predicted_label = np.argmax(predicted_label)
            else:
                predicted_label = predicted_label[0]

        x_train, y = self.data
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        ts_length = self.ts_length
        num_feature = self.num_feature
        knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
        knn.fit(x_train[list(np.where(y == target))].reshape(-1, ts_length, num_feature))
        dist, ind = knn.kneighbors(query.reshape(1, ts_length, num_feature), return_distance=True)
        x_train.reshape(-1, 1, ts_length)
        return dist[0], x_train[np.where(y == target)][ind[0]]

    def _nearest_unlike_neighbor(self, query, predicted_label,target, distance, n_neighbors):
        _, nun = self._nearest_unlike_retrieval(
            query, predicted_label,target, distance, n_neighbors
        )
        if nun is None:

            return None, None
        individual = np.array(nun.tolist())  # , dtype=np.float64)
        out = self.predict(individual)
        if np.argmax(out) == predicted_label:
            print(out, predicted_label,target)
            orig_pred = self.predict(query)
            print(orig_pred,np.argsort(orig_pred)[0][-2:-1][0],np.argsort(orig_pred),np.argmax(orig_pred))
            print("No Counterfactual found. Most likely caused by a constant predictor.")
            return None, None
        if np.argmax(out) != target:
            print(out, target)
            print("Not a correct target label.")
            return None, None

        return nun, np.argmax(out)

    def _findSubarray(
            self, a, k
    ):  # used to find the maximum contigious subarray of length k in the explanation weight vector
        if len(a.shape) == 2:
            a = a.reshape(-1)
        n = len(a)
        a = a.tolist()

        vec = []

        # Iterate to find all the sub-arrays
        for i in range(n - k + 1):
            temp = []

            # Store the sub-array elements in the array
            for j in range(i, i + k):
                temp.append(a[j])

            # Push the vector in the container
            vec.append(temp)
        sum_arr = []
        for v in vec:
            sum_arr.append(np.sum(v))

        return np.argmax(sum_arr), vec[np.argmax(sum_arr)]

    # def _counterfactual_generator_swap(
    #     self, instance, label, subarray_length=1, max_iter=500
    # ):
    def _counterfactual_generator_swap(
            self, instance, label,target, subarray_length=1, max_iter=500
    ):
        _, nun = self._nearest_unlike_retrieval(instance, label, target, self.distance_measure, 1)
        if np.count_nonzero(nun.reshape(-1) - instance.reshape(-1)) == 0:
            print("Starting and nun are Identical !")

        test_x, test_y = self.data
        train_x = test_x
        individual = np.array(nun.tolist())  # , dtype=np.float64)
        out = self.predict(individual)
        if self.backend == "PYT":
            training_weights = (
                self.cam_extractor(out.squeeze(0).argmax().item(), out)[0]
                .detach().cpu()
                .numpy()
            )
        elif self.backend == "TF":
            data = (instance.reshape(1, -1, 1), None)
            training_weights = self.cam_extractor.explain(
                data, self.model, class_index=label[0]
            )  # grad_cam(self.model, instance.reshape(1,-1,1))#self.cam_extractor.explain(data, self.model,class_index=label)#instance
        # Classify Original
        individual = np.array(instance.tolist())  # , dtype=np.float64)
        out = self.predict(individual)

        starting_point, most_influencial_array = self._findSubarray((training_weights), subarray_length)

        if np.any(np.isnan(most_influencial_array)):
            return np.full(individual.shape, None), None

        # starting_point = np.where(training_weights == most_influencial_array[0])[0][0]

        X_example = instance.copy().reshape(1, -1)

        nun = nun.reshape(1, -1)
        X_example[0, starting_point: subarray_length + starting_point] = nun[
                                                                         0, starting_point: subarray_length + starting_point
                                                                         ]
        individual = np.array(
            X_example.reshape(-1, 1, train_x.shape[-1]).tolist()
        )  # , dtype=np.float64
        # )
        out = self.predict(individual)
        prob_target = out[0][
            label
        ]  # torch.nn.functional.softmax(model(torch.from_numpy(test_x))).detach().numpy()[0][y_pred[instance]]

        counter = 0
        pred = np.argmax(out, axis=1)[0]
        #while prob_target > 0.5 and counter < max_iter: # Ziwen This is not a valid way to flip the model, chances are that the original class is<0.5 but still the largest probability
        while pred != target and counter < max_iter:
            subarray_length += 1
            starting_point, most_influencial_array = self._findSubarray(
                (training_weights), subarray_length
            )
            # starting_point = np.where(training_weights == most_influencial_array[0])[0][
            #     0
            # ]
            X_example = instance.copy().reshape(1, -1)
            X_example[:, starting_point: subarray_length + starting_point] = nun[
                                                                             :, starting_point: subarray_length + starting_point
                                                                             ]
            individual = np.array(
                X_example.reshape(-1, 1, train_x.shape[-1]).tolist()
            )  # , dtype=np.float64
            # )
            out = self.predict(individual)
            prob_target = out[0][label]
            pred = np.argmax(out, axis=1)[0]
            counter = counter + 1
            # if counter == max_iter or subarray_length == self.ts_length:
            if counter == max_iter:
                print("No Counterfactual found")
                return None, None
        if np.argmax(out)!= target:
            print(np.argmax(out), target,label)
            print("Not a correct target label.")
            return None, None

        return X_example, np.argmax(out, axis=1)[0]

    def _instance_based_cf(self, query, label, target, distance="dtw", max_iter=500):
        d, nan = self._nearest_unlike_retrieval(query, label, distance, 1)
        beta = 0
        insample_cf = nan.reshape(1, 1, -1)

        individual = np.array(query.tolist())  # , dtype=np.float64)

        output = self.predict(individual)
        pred_treshold = 0.5
        target = np.argsort(output)[0][-2:-1][0]
        query = query.reshape(-1)
        insample_cf = insample_cf.reshape(-1)
        generated_cf = dtw_barycenter_averaging(
            [query, insample_cf], weights=np.array([(1 - beta), beta])
        )
        generated_cf = generated_cf.reshape(1, 1, -1)
        individual = np.array(generated_cf.tolist())  # , dtype=np.float64)
        prob_target = self.predict(individual)[0][target]
        counter = 0

        while prob_target < pred_treshold and counter < max_iter:
            beta += 0.01
            generated_cf = dtw_barycenter_averaging(
                [query, insample_cf], weights=np.array([(1 - beta), beta])
            )
            generated_cf = generated_cf.reshape(1, 1, -1)
            individual = np.array(generated_cf.tolist())  # , dtype=np.float64)
            prob_target = self.predict(individual)[0][target]

            counter = counter + 1
        if counter == max_iter:
            print("No Counterfactual found")
            return None, None

        return generated_cf, target

    def explain(self, x: np.ndarray, y: int) -> Tuple[np.array, int]:
        """'
        Explains a specific instance x.
        Arguments:
            x np.array : instance to be explained.
            y int: predicted label for instance x.
        Returns:
            Tuple: (counterfactual, counterfactual label)

        """
        if self.mode == "time":
            x = x.reshape(x.shape[0], x.shape[2], x.shape[1])

        output = self.predict(x)
        target = np.argsort(output)[0][-2:-1][0]

        if self.method == "NUN_CF":
            return self._nearest_unlike_neighbor(
                x, y,target, self.distance_measure, self.n_neighbors
            )
        elif self.method == "dtw_bary_center":
            return self._instance_based_cf(x, y, self.distance_measure)
        elif self.method == "NG":

            self.distance_measure = "euclidean"
            return self._counterfactual_generator_swap(x, y,target, max_iter=self.max_iter)
        else:
            print("Unknown Method selected.")
