import os.path

import torch
from torch.nn import ReLU

# from models.ResNet import ResNetBaseline
# from models.FCN import FCN
from models.FCN import FCN_AE
from tsai.models.InceptionTime import InceptionTime
from tsai.models.MLP import MLP
from tsai.models.FCN import FCN
from tsai.models.ResNet import ResNet

def model_init(model_name, in_channels, n_pred_classes, seq_len=None):
    # n_pred_classes = train_y.shape[1]
    # Those models don't have softmax as nn.CrossEntropyLoss alreadyhas SoftMax insie
    if model_name == 'ResNet':
        # model = ResNetBaseline(in_channels=in_channels, num_pred_classes=n_pred_classes)
        model = ResNet(c_in=in_channels, c_out=n_pred_classes)
    elif model_name == 'FCN':
        # model = FCN(input_shape=in_channels, nb_classes=n_pred_classes)
        model = FCN(c_in=in_channels, c_out=n_pred_classes)
    elif model_name == 'FCN_AE':
        model = FCN_AE(input_shape=in_channels, nb_classes=n_pred_classes)
    elif model_name == 'InceptionTime':
        model = InceptionTime(in_channels, n_pred_classes)
    elif model_name == 'MLP':
        model = MLP(c_in=in_channels, c_out=n_pred_classes, seq_len=seq_len, layers=[500, 500, 500], ps=[0.1, 0.2, 0.2],
                    act=ReLU(inplace=True))
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
