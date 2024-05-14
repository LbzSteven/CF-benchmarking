from typing import cast, Dict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


def fit(model, train_loader, device, num_epochs: int = 1500,
        learning_rate: float = 0.001,
        patience: int = 100, ) -> None:  # patience war 10

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_train_loss = np.inf
    patience_counter = 0
    best_state_dict = None

    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_train_loss = 0
        for x_t, y_t in train_loader:
            x_t, y_t = x_t.to(device), y_t.to(device)
            optimizer.zero_grad()
            output = model(x_t.float())
            if len(y_t.shape) == 1:
                train_loss = F.binary_cross_entropy_with_logits(
                    output, y_t.unsqueeze(-1).float(), reduction='mean'
                )
            else:
                train_loss = F.cross_entropy(output, y_t.argmax(dim=-1), reduction='mean')

            epoch_train_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()

        model.eval()

        if epoch_train_loss < best_train_loss:
            best_train_loss = epoch_train_loss
            best_state_dict = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter == patience:
                if best_state_dict is not None:
                    model.load_state_dict(cast(Dict[str, torch.Tensor], best_state_dict))
                print(f'Early stopping! at {epoch + 1}, using state at {epoch + 1 - patience}')
                return None


def get_all_preds(model, loader, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        all_preds = []
        labels = []
        for batch in loader:
            item, label = batch
            item = item.to(device)
            preds = model(item.float())
            all_preds = all_preds + preds.cpu().argmax(dim=1).tolist()
            labels = labels + label.tolist()
    return all_preds, labels


class UCRDataset(Dataset):
    def __init__(self, feature, target):
        self.feature = feature
        self.target = target

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        item = self.feature[idx]
        label = self.target[idx]

        return item, label


def generate_loader(train_x, test_x, train_y, test_y, batch_size_train=16, batch_size_test=1):
    train_dataset = UCRDataset(train_x.astype(np.float64), train_y.astype(np.int64))
    test_dataset = UCRDataset(test_x.astype(np.float64), test_y.astype(np.int64))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    return train_loader, test_loader


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class Conv1dSamePadding(nn.Conv1d):

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


# For Autoencoder:
def fit_AE(model, train_loader, device, num_epochs: int = 200, learning_rate: float = 0.001,
           patience: int = 10) -> None:  # patience was 10

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    best_train_loss = np.inf
    patience_counter = 0
    best_state_dict = None

    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_train_loss = []
        for x_t, y_t in train_loader:
            train_loss_total = 0
            x_t, _ = x_t.to(device), y_t.to(device)
            optimizer.zero_grad()
            output = model(x_t.float())

            train_loss = criterion(output, x_t.float())
            train_loss.backward()
            optimizer.step()
            train_loss_total += train_loss.item()
        if train_loss_total < best_train_loss:
            best_train_loss = train_loss_total
            best_state_dict = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter == patience:
                if best_state_dict is not None:
                    model.load_state_dict(cast(Dict[str, torch.Tensor], best_state_dict))
                print(f'Early stopping! at {epoch + 1}, using state at {epoch + 1 - patience}, best loss {best_train_loss:.2f}')
                return None




# get loss of a whole datasets.
def get_loss(model, data, label, device):
    model.to(device)
    model.eval()
    criterion = torch.nn.MSELoss()
    dataset = UCRDataset(data.astype(np.float64), label.astype(np.int64))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        all_preds = []
        all_loss = []
        for batch in loader:
            item, _ = batch
            item = item.to(device)
            preds = model(item.float())
            all_preds.append(preds.cpu().detach().numpy().reshape(data.shape[-2], -1))
            loss = criterion(preds, item)
            all_loss.append(loss.item())
        all_preds = np.array(all_preds)
        all_loss = np.array(all_loss)
    return all_preds, all_loss
