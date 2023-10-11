import torch
import torch.nn as nn
import torch.nn.functional as F
import torch


def calc_conv_out_feat(W, K=[], P=[], S=[]):
    out_W = W
    for i in range(len(K)):
        out_W = ((out_W - K[i] + 2 * P[i]) / S[i]) + 1
    return out_W


class CNN(nn.Module):
    def __init__(self, num_class=7, checkpoint_file='./tmp/'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        # fc1_in_feat = calc_conv_out_feat(im_size[0],K=[5,2,5,2], P=[0,0,0,0], S=[1,2,1,2])
        self.fc1 = nn.Linear(2704, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

        self.class_num = num_class

        self.checkpoint_file = checkpoint_file

    def forward(self, x):
        shape_length = len(x.shape)
        if shape_length == 2 + 3:
            batch_length = x.shape[0]
            batch_size = x.shape[1]
            x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        elif shape_length == 3 + 3:
            num_samples = x.shape[0]
            batch_length = x.shape[1]
            batch_size = x.shape[2]
            x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        elif shape_length == 3:
            x = x.unsqueeze(0)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if shape_length == 2 + 3:
            x = x.reshape(batch_length, batch_size, -1)
        elif shape_length == 3 + 3:
            x = x.reshape(num_samples, batch_length, batch_size, -1)
        elif shape_length == 3:
            x = x.squeeze(0)
        return x

    def get_prediction(self, x):
        x_tmp = self.forward(x)
        pred_prob = torch.softmax(x_tmp, dim=-1)
        pred = torch.argmax(pred_prob, dim=-1)

        return pred, pred_prob

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))  # + episode))


class XClassifier(torch.nn.Module):
    def __init__(self, model_info_dict={}, save_path='./tmp/'):
        super().__init__()
        self.model_dict = torch.nn.ModuleDict()
        self.model_info_dict = model_info_dict

        for model_name, model_info in model_info_dict.items():
            if model_info['type'] == 'image':
                self.model_dict[model_name] = CNN(num_class=model_info['class_num'],
                                                  checkpoint_file=save_path + model_name)
            else:
                raise NotImplementedError

    def forward(self, x_dict):
        """

        :param x_dict: {'data', 'type'}
        :return:
        """
        pred_dict = {}
        for x, x_name in x_dict.items():
            pred_data = self.model_dict[x_name](x['data'])
            pred_dict[x_name] = pred_data

        return pred_dict

    def get_prediction(self, x_dict):
        pred_dict = {}
        for x, x_name in x_dict.items():
            pred, pred_prob = self.model_dict[x_name].get_prediction(x['data'])
            pred_dict[x_name]['pred'] = pred
            pred_dict[x_name]['pred_prob'] = pred_prob
        return pred_dict

    def save_checkpoint(self):
        for model_name in self.model_info_dict:
            self.model_dict[model_name].save_checkpoint()

    def load_checkpoint(self):
        for model_name in self.model_info_dict:
            self.model_dict[model_name].load_checkpoint()

class XRegression(torch.nn.Module):
    def __init__(self, model_info_dict={}, save_path='./tmp/', device='0'):
        super().__init__()
        self.model_dict = torch.nn.ModuleDict()
        self.model_info_dict = model_info_dict

        for model_name, model_info in model_info_dict.items():
            if model_info['type'] == 'image':
                self.model_dict[model_name] = TableAssemblyRegressionModel(num_class=model_info['class_num'],
                                                  checkpoint_file=save_path + model_name, device=device)
            else:
                raise NotImplementedError

    def forward(self, x_dict):
        """

        :param x_dict: {'data', 'type'}
        :return:
        """
        pred_dict = {}
        for x, x_name in x_dict.items():
            pred_data = self.model_dict[x_name](x['data'])
            pred_dict[x_name] = pred_data

        return pred_dict

    def get_prediction(self, x_dict):
        pred_dict = {}
        for x, x_name in x_dict.items():
            pred, pred_prob = self.model_dict[x_name].get_prediction(x['data'])
            pred_dict[x_name]['pred'] = pred
            pred_dict[x_name]['pred_prob'] = pred_prob
        return pred_dict

    def save_checkpoint(self):
        for model_name in self.model_info_dict:
            self.model_dict[model_name].save_checkpoint()

    def load_checkpoint(self):
        for model_name in self.model_info_dict:
            self.model_dict[model_name].load_checkpoint()

class TableAssemblyRegressionModel(nn.Module):
    def __init__(self,image_embed_size=48,linear_embed_input=1024 + 32, device='0',num_class=1, checkpoint_file=None):
        super().__init__()
        self.checkpoint_file = checkpoint_file
        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(3, 16, 4, stride=2),
            conv2d_bn_relu(16, 16, 3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(16, 32, 4, stride=2),
            conv2d_bn_relu(32, 32, 3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(32, 64, 4, stride=2),
            conv2d_bn_relu(64, 64, 3)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(64, 64, 4, stride=2),
            conv2d_bn_relu(64, 64, 3),
        )

        stride = 1
        self.conv_spatial_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(3, 32, 4, stride=stride),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        )
        self.conv_spatial_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(32, 16, 4, stride=stride),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        )
        self.device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
        self.spatial_conv = SpatialSoftmaxLayer(n_channel=16, n_rows=62, n_cols=62, device=self.device)

        self.linear_embed = torch.nn.Sequential(
            nn.Linear(linear_embed_input, 2 * image_embed_size),
            nn.ReLU(),
            nn.Linear(2 * image_embed_size, num_class),
        )
        self.to(self.device)

    def forward(self, x):
        shape_length = len(x.shape)
        if shape_length == 2 + 3:
            batch_length = x.shape[0]
            batch_size = x.shape[1]
            x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        elif shape_length == 3 + 3:
            num_samples = x.shape[0]
            batch_length = x.shape[1]
            batch_size = x.shape[2]
            x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        elif shape_length == 3:
            x = x.unsqueeze(0)

        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)

        conv1_spatial_out = self.conv_spatial_stack1(x)
        conv2_spatial_out = self.conv_spatial_stack2(conv1_spatial_out)
        sptial_out = self.spatial_conv(conv2_spatial_out).detach()

        cnn_feature = torch.cat([conv4_out.reshape(x.shape[:-3] + (-1,)),
                                 sptial_out], dim=-1)
        # # cnn_feature = conv4_out.reshape(x.shape[:-3] + (-1,))
        # cnn_feature = sptial_out
        linear_out = self.linear_embed(cnn_feature)

        if shape_length == 2 + 3:
            linear_out = linear_out.reshape(batch_length, batch_size, -1)
        elif shape_length == 3 + 3:
            linear_out = linear_out.reshape(num_samples, batch_length, batch_size, -1)
        elif shape_length == 3:
            linear_out = linear_out.squeeze(0)
        return linear_out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))  # + episode))

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

def conv2d_bn_relu(inch, outch, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch, outch, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer
