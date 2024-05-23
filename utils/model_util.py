import os.path

import torch

from models.ResNet import ResNetBaseline
from models.FCN import FCN, FCN_AE
from tsai.models.InceptionTime import InceptionTime

def model_init(model_name, in_channels, n_pred_classes):
    # n_pred_classes = train_y.shape[1]
    if model_name == 'ResNet':
        model = ResNetBaseline(in_channels=in_channels, num_pred_classes=n_pred_classes)
    elif model_name == 'FCN':
        model = FCN(input_shape=in_channels, nb_classes=n_pred_classes)
    elif model_name == 'FCN_AE':
        model = FCN_AE(input_shape=in_channels, nb_classes=n_pred_classes)
    elif model_name == 'InceptionTime':
        model = InceptionTime(in_channels, n_pred_classes)

    else:
        raise 'Wrong model'
    return model


def AE_init(model_name, in_channels, input_size):
    if model_name == 'FCN_AE':
        model = FCN_AE(input_shape=in_channels, input_size=input_size)
    else:
        raise 'Wrong model'
    return model


def get_AE_dict(model_name, path, in_channels, input_size):
    AE_dict = {}
    AE = AE_init(model_name, in_channels, input_size)
    state_dict = torch.load(f'{path}/AE')
    AE.load_state_dict(state_dict)
    AE.eval()
    AE_dict['AE'] = AE
    class_specific_AEs = [p for p in os.listdir(path) if 'AE_' in p]
    print(class_specific_AEs)
    for name in class_specific_AEs:
        c = int(name.split('_')[-1])
        AEc = AE_init(model_name, in_channels, input_size)
        state_dict = torch.load(f'{path}/AE_{c}')
        AEc.load_state_dict(state_dict)
        AEc.eval()
        AE_dict[c] = AEc
    return AE_dict


