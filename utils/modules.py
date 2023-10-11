import torch
import torch.nn as nn

def conv2d_bn_leakrelu(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.LeakyReLU()
    )
    return convlayer


def conv2d_bn_relu(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer


def deconv_tanh(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.Tanh()
    )
    return convlayer


def deconv_sigmoid(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.Sigmoid()
    )
    return convlayer


def deconv_leakrelu(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.LeakyReLU()
    )
    return convlayer


def deconv_relu(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer


class SpatialSoftmaxLayer(nn.Module):
    """
    C, W, H => C*2 # for each channel, get good point [x,y]
    """

    def __init__(self, n_channel, n_rows, n_cols, device="cuda:0"):
        super(SpatialSoftmaxLayer, self).__init__()

        x_map = torch.zeros((n_channel, n_rows, n_cols))
        y_map = torch.zeros((n_channel, n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                x_map[:, i, j] = (i - n_rows / 2) #/ n_rows
                y_map[:, i, j] = (j - n_cols / 2) #/ n_cols

        self.x_map = x_map.detach().to(device)
        self.y_map = y_map.detach().to(device)

        self.SoftMax = torch.nn.Softmax(dim=-1)

    def forward(self, x):  # after relu
        batch, n_channel, width, height = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])  # batch, C, W*H
        x = self.SoftMax(x)  # batch, C, W*H
        x = x.view(batch, n_channel, width, height)
        fp_x = (x * self.x_map).sum(dim=-1).sum(dim=-1)  # batch, C
        fp_y = (x * self.y_map).sum(dim=-1).sum(dim=-1)  # batch, C
        x = torch.cat((fp_x, fp_y), dim=1)
        return x  # batch, C*2