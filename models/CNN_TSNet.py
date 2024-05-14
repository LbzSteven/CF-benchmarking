import torch
from torch import nn
class CNN_TSNet(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0, input_size=0, output=2, out_channels=2):
        super(CNN_TSNet, self).__init__()
        # in (batch_size, in_channels, length)
        # out (batch_size, out_channels, (lenght - kernel + 1))
        len_in = input_size
        len_out = ((len_in + (2 * padding) - (kernel_size - 1) - 1) / stride) + 1
        # print('len',len_out)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                padding=padding)
        # activation
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(int(len_out * out_channels), 50)  # len_out * out_channels
        # TODO Flex
        self.fc2 = nn.Linear(50, output)

        self.gradients = None

    def forward(self, x):
        x = torch.Tensor(x)

        x = self.conv1d(x)
        if self.train and x.requires_grad:
            h = x.register_hook(self.activations_hook)
        x = self.relu(x)

        x = x.flatten(start_dim=1)
        # x = x.view(-1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.conv1d(x)