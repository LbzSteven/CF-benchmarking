import torch
from torch import nn

from utils.train_util import fit, Conv1dSamePadding


class ResNetBaseline(nn.Module):
    """A PyTorch implementation of the ResNet Baseline
    From https://arxiv.org/abs/1909.04939
    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    mid_channels:
        The 3 residual blocks will have as output channels:
        [mid_channels, mid_channels * 2, mid_channels * 2]
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, mid_channels: int = 64,
                 num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        self.layers = nn.Sequential(*[
            ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
            ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
            ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),

        ])
        self.final = nn.Linear(mid_channels * 2, num_pred_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.layers(x)
        return self.final(x.mean(dim=-1))


class ResNetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
                      kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)


import torch
from torch import nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)


if __name__ == '__main__':
    import pandas as pd
    import os
    import numpy as np
    from pathlib import Path
    import platform
    import sklearn
    from CNN_TSNet import UCRDataset

    os_type = platform.system()
    dataset = 'SmallKitchenAppliances'  # 'ArrowHead'
    if os_type == 'Linux':
        # print(Path('/media/jacqueline/Data/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv'))
        train = pd.read_csv(Path(f'/media/jacqueline/Data/UCRArchive_2018/{dataset}/{dataset}_TRAIN.tsv'), sep='\t',
                            header=None)
        test = pd.read_csv(os.path.abspath(f'/media/jacqueline/Data/UCRArchive_2018/{dataset}/{dataset}_TEST.tsv'),
                           sep='\t', header=None)
    train_y, train_x = train.loc[:, 0].apply(lambda x: x - 1).to_numpy(), train.loc[:, 1:].to_numpy()
    test_y, test_x = test.loc[:, 0].apply(lambda x: x - 1).to_numpy(), test.loc[:, 1:].to_numpy()
    # print(train_y.reshape(-1,1))
    enc1 = sklearn.preprocessing.OneHotEncoder(sparse=False).fit(train_y.reshape(-1, 1))
    # enc2=sklearn.preprocessing.OneHotEncoder().fit_transform(test_y.reshape(-1,1))
    # print(sklearn.preprocessing.OneHotEncoder().fit_transform(train_y.astype(str).reshape(-1,1)))
    # print(train_y.shape)
    train_y = enc1.transform(train_y.reshape(-1, 1))
    test_y = enc1.transform(test_y.reshape(-1, 1))

    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    # print(train_y)
    train_dataset = UCRDataset(train_x.astype(np.float64), train_y.astype(np.int64))
    test_dataset = UCRDataset(test_x.astype(np.float64), test_y.astype(np.int64))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    # model=ResNet(output=3)
    # train_2(train_loader,test_loader, model,100)
    model = ResNetBaseline(in_channels=1, num_pred_classes=3)
    fit(model, train_loader, test_loader)
    torch.save(model.state_dict(), 'test')
    pass