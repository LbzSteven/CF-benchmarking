import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_target_1d(input_tensor, target_length):
    # Calculate padding
    print(input_tensor.shape[2], target_length)
    padding_length = target_length - input_tensor.shape[2]

    # Ensure padding is non-negative
    if padding_length < 0:
        raise ValueError("Target length must be greater than or equal to input length")

    # Calculate padding for each side: (left, right)
    pad_left = padding_length // 2
    pad_right = padding_length - pad_left

    # Apply padding
    padded_tensor = F.pad(input_tensor, (pad_left, pad_right))

    return padded_tensor


class Encoder(nn.Module):
    def __init__(self, input_shape, dropout):
        super(Encoder, self).__init__()

        self.dropout = dropout

        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=128, kernel_size=5, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=11, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(256)
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=21, stride=1, padding='same')
        self.bn3 = nn.BatchNorm1d(512)
        self.prelu3 = nn.PReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.dropout1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.dropout2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        x = self.dropout3(x)

        attention_data = x[:, :256, :]
        attention_softmax = self.softmax(x[:, 256:, :])
        multiply_layer = attention_softmax * attention_data

        flatten = multiply_layer.view(multiply_layer.size(0), -1)

        return flatten


class Decoder(nn.Module):
    def __init__(self, encoded_shape, original_shape, dropout):
        super(Decoder, self).__init__()

        self.dropout = dropout

        self.reshape = (-1, 256)

        self.deconv1 = nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=21, stride=1, padding=10)
        self.bn1 = nn.BatchNorm1d(256)
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.deconv2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=11, stride=1, padding=5)
        self.bn2 = nn.BatchNorm1d(128)
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.deconv3 = nn.ConvTranspose1d(in_channels=128, out_channels=original_shape[0], kernel_size=5, stride=1, padding=2)
        self.original_shape = original_shape

    def forward(self, x):
        x = x.view(x.size(0), 256, -1)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.dropout1(x)
        x = self.upsample1(x)
        print(x.shape[2], self.original_shape[1] // 2)
        if x.shape[2] != self.original_shape[1] // 2:
            x = pad_to_target_1d(x, self.original_shape[1] // 2)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.dropout2(x)
        x = self.upsample2(x)
        if x.shape[2] != self.original_shape[1]:
            x = pad_to_target_1d(x, self.original_shape[1])
        x = self.deconv3(x)

        return x


class Autoencoder(nn.Module):
    def __init__(self, input_shape, dropout):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_shape, dropout)
        encoded_shape = (256, input_shape[1] // 4)  # Adjust based on pooling layers
        self.decoder = Decoder(encoded_shape, input_shape, dropout)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

