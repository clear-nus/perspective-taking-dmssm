import torch
from utils.modules import conv2d_bn_relu, deconv_tanh, deconv_relu, deconv_sigmoid, SpatialSoftmaxLayer
import numpy as np
import os
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils.tools import *
from torch.nn.modules.activation import MultiheadAttention
from typing import Tuple, Optional
from torch.nn import functional as F


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayerMod(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayerMod, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayerMod, self).__setstate__(state)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        # src1 = src1 + self.dropout1(src1)
        src1 = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src1))))
        src3 = src + self.dropout2(src2)
        src3 = self.norm2(src3)
        return src3


class TransformerSimpleNetwork(nn.Module):
    def __init__(self, input_dims: tuple, output_dims, trans_hidden_dims: int,
                 ff_hidden_dims: int,
                 batch_length=5, num_heads=3, layers=1,
                 dropout=0.0, activation=nn.ELU, min=1e-4, max=10.0, name='Transformer', chkpt_dir='tmp', device='0'):
        '''
        Transformer takes in input of shape (target_length, batch_size, embed_dim)
        Each head will attend to embed_dim//num_heads number of dimension

        input_dims: input dimension into the Transformer Embedding layer (s_size + action_size)
        trans_hidden_dims: hidden dims used in the transfomer (must be divisible by num_heads)
        output_dims: output dimension (h_size) of the final decoder layer
        ff_hidden_dims : hidden dims in feedforwardnetwork
        layers: number of layers of MHA + feedforward network
        '''
        super().__init__()
        # For adjusting pytorch to tensorflow
        self._feature_size = input_dims[0]
        self.output_dims = output_dims[0]
        if trans_hidden_dims // num_heads == 0:
            self.trans_hidden_dims = trans_hidden_dims
        else:
            self.trans_hidden_dims = num_heads * 6
        self._hidden_size = ff_hidden_dims
        self.batch_length = batch_length
        self.num_heads = num_heads
        self._layers = layers
        self.dropout = dropout
        self.activation = activation

        # Building Transformer Encoder
        self.encoder = nn.Linear(self._feature_size, self.trans_hidden_dims)
        self.pos_encoder = PositionalEncoding(self.trans_hidden_dims, self.dropout)
        encoder_layers = TransformerEncoderLayer(d_model=self.trans_hidden_dims, nhead=self.num_heads,
                                                 dim_feedforward=self._hidden_size, dropout=self.dropout)
        self.transformer_first_encoder = TransformerEncoderLayerMod(d_model=self.trans_hidden_dims, nhead=self.num_heads,
                                                                    dim_feedforward=self._hidden_size, dropout=self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self._layers)
        self.decoder = nn.Linear(self.trans_hidden_dims, self.output_dims)
        self.init_weights()

        # Attributes for Distribution
        self.soft_plus = nn.Softplus()
        self._min = min
        self._max = max

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask, src_padding_mask=None):
        '''
        Transformer takes in input of shape (target_length, batch_size, embed_dim)
        '''
        shape_len = len(src.shape)
        if shape_len == 3:
            batch = src.shape[1]
            length = src.shape[0]
            # features = features.reshape(-1, features.shape[-1])
        src = self.encoder(src) * math.sqrt(self.trans_hidden_dims)
        src = self.pos_encoder(src)
        src = self.transformer_first_encoder(src, src_mask, src_key_padding_mask=src_padding_mask)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask=src_padding_mask)
        dist_inputs = self.decoder(output)
        return dist_inputs

    def save_checkpoint(self, episode):
        torch.save(self.state_dict(), self.checkpoint_file + episode)

    def load_checkpoint(self, episode, best=False):
        if not best:
            self.load_state_dict(torch.load(self.checkpoint_file + episode))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file + '_best'))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        # x = torch.zeros_like(x).to(x.device) + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiCategoricalDist:
    def __init__(self, inputs, num_cat=1, if_prob=False):
        super().__init__()
        self.num_cat = num_cat
        self.feature_size = inputs.shape[-1]
        self.batch_size = inputs.shape[:-1]

        cat_feature_size = int(self.feature_size / num_cat)
        inputs_reshaped = inputs.reshape(self.batch_size + (num_cat, cat_feature_size))

        if if_prob:
            self.dist = torch.distributions.independent.Independent(
                torch.distributions.OneHotCategorical(
                    probs=inputs_reshaped), 1)
        else:
            self.dist = torch.distributions.independent.Independent(
                torch.distributions.OneHotCategorical(
                    logits=inputs_reshaped), 1)

        self.mean = self.mean()
        self.stddev = self.stddev()

    def sample(self, size=None):
        if size is None:
            sample = self.dist.sample().reshape(self.batch_size + (self.feature_size,))
        else:
            sample = self.dist.sample(size).reshape(size + self.batch_size + (self.feature_size,))
        return sample

    def rsample(self):
        rsample = self.dist.sample().reshape(self.batch_size + (self.feature_size,))
        rsample += self.dist.mean.reshape(self.batch_size + (self.feature_size,))
        rsample -= self.dist.mean.reshape(self.batch_size + (self.feature_size,)).detach()
        return rsample

    def mean(self):
        mean = self.dist.mean.reshape(self.batch_size + (self.feature_size,))
        return mean

    def stddev(self):
        stddev = self.dist.stddev.reshape(self.batch_size + (self.feature_size,))
        return stddev


class PerspTakingBaselineNetwork(nn.Module):
    def __init__(self, chkpt_dir='tmp', name='BaselinePerspTaking', device='0'):
        super().__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name)

        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(3, 32, 4, stride=2),
            conv2d_bn_relu(32, 32, 3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(32, 32, 4, stride=2),
            conv2d_bn_relu(32, 32, 3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(32, 64, 4, stride=2),
            conv2d_bn_relu(64, 64, 3)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(64, 128, 4, stride=2),
            conv2d_bn_relu(128, 128, 3),
        )

        self.deconv_4 = deconv_relu(128, 64, 4, stride=2)
        self.deconv_3 = deconv_relu(67, 32, 4, stride=2)
        self.deconv_2 = deconv_relu(35, 16, 4, stride=2)
        self.deconv_1 = deconv_sigmoid(19, 3, 4, stride=2)

        self.predict_4 = torch.nn.Conv2d(128, 3, 3, stride=1, padding=1)
        self.predict_3 = torch.nn.Conv2d(67, 3, 3, stride=1, padding=1)
        self.predict_2 = torch.nn.Conv2d(35, 3, 3, stride=1, padding=1)

        self.up_sample_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1, bias=False),
            torch.nn.Sigmoid()
        )
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        """inputimag: BS,4,3,H,W

               x-->conv_stack1 --> conv_stack2 --> conv_stack3 -->  conv_stack4-->              deconv_4     --->           deconv_3        -->       deconv_2            -->deconv_1 --> output
                                                                                     \--> predict_4>up_sample_4---/\--> predict_3>up_sample_3---/\--> predict_2>up_sample_2---/
        """

        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)

        deconv4_out = self.deconv_4(conv4_out)
        predict_4_out = self.up_sample_4(self.predict_4(conv4_out))

        concat_4 = torch.cat([deconv4_out, predict_4_out], dim=1)
        deconv3_out = self.deconv_3(concat_4)
        predict_3_out = self.up_sample_3(self.predict_3(concat_4))

        concat2 = torch.cat([deconv3_out, predict_3_out], dim=1)
        deconv2_out = self.deconv_2(concat2)
        predict_2_out = self.up_sample_2(self.predict_2(concat2))

        concat1 = torch.cat([deconv2_out, predict_2_out], dim=1)
        predict_out = self.deconv_1(concat1)

        return predict_out

    def save_checkpoint(self):
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, best=False):
        if not best:
            self.load_state_dict(torch.load(self.checkpoint_file))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file + '_best'))

class PerspTakingNetwork(nn.Module):
    def __init__(self, agent_obs_info_dict, pi_info_dict, name='pt_model', embed_size=32, pos_size=1, device='0'):
        super().__init__()
        self.image_embed_model = MultiModalEmbed(modality_info_dict=pi_info_dict,
                                                 embed_size=embed_size,
                                                 name=name + 'pt_embed', device=device)

        self.fuse_model = SimpleMLP(name=name + 'fuse',
                                    input_dims=(embed_size + pos_size,),
                                    hidden_dims=64,
                                    output_dims=(embed_size,), layers=2,
                                    device=device)

        self.image_decode_model = MultiModalDecoder(modality_info_dict=agent_obs_info_dict, latent_size=embed_size,
                                                    name=name + 'pt_decode', device=device)

    def forward(self, pi_dict, pos):
        image_embed = self.image_embed_model(pi_dict)
        fuse_feature = torch.cat([image_embed, pos], dim=-1)
        fuse_feature = self.fuse_model(fuse_feature)
        image_decode = self.image_decode_model(fuse_feature)
        return image_decode

    def save_checkpoint(self, episode):
        self.image_embed_model.save_checkpoint(episode)
        self.fuse_model.save_checkpoint(episode)
        self.image_decode_model.save_checkpoint(episode)

    def load_checkpoint(self, episode, best=False):
        if not best:
            self.image_embed_model.load_checkpoint(episode)
            self.fuse_model.load_checkpoint(episode)
            self.image_decode_model.load_checkpoint(episode)
        else:
            self.image_embed_model.load_checkpoint(episode, best=True)
            self.fuse_model.load_checkpoint(episode, best=True)
            self.image_decode_model.load_checkpoint(episode, best=True)

class ImageEmbedCNN(torch.nn.Module):
    """
    Input image dimensions: [batchSize, Channel, Height, Width]
    Input image shape: [64, 3, 64, 64]
    output image shape: [64, 128, 4, 4]
    """

    def __init__(self, image_embed_size=48,
                 linear_embed_input=1024 + 32,  # 2048 for 64x64 and 8192 for 128x128
                 chkpt_dir='tmp',
                 name='encoder',
                 device='0'):
        super(ImageEmbedCNN, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name)
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

        self.spatial_conv = SpatialSoftmaxLayer(n_channel=16, n_rows=62, n_cols=62, device=f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

        self.linear_embed = torch.nn.Sequential(
            nn.Linear(linear_embed_input, 2 * image_embed_size),
            nn.ReLU(),
            nn.Linear(2 * image_embed_size, image_embed_size),
        )
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
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

    def save_checkpoint(self, episode=''):
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
        torch.save(self.state_dict(), self.checkpoint_file + episode)

    def load_checkpoint(self, episode='', best=False):
        if not best:
            self.load_state_dict(torch.load(self.checkpoint_file))  # + episode))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file + episode + '_best'))


class ImageEmbedCNNHighResol(torch.nn.Module):
    """
    Input image dimensions: [batchSize, Channel, Height, Width]
    Input image shape: [64, 3, 64, 64]
    output image shape: [64, 128, 4, 4]
    """

    def __init__(self, image_embed_size=48, chkpt_dir='tmp', name='encoder', device='0'):
        super(ImageEmbedCNNHighResol, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(3, 32, 4, stride=2),
            conv2d_bn_relu(32, 32, 3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(32, 32, 4, stride=2),
            conv2d_bn_relu(32, 32, 3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(32, 64, 4, stride=2),
            conv2d_bn_relu(64, 64, 3)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(64, 128, 4, stride=2),
            conv2d_bn_relu(128, 128, 3),
        )
        self.linear_embed = torch.nn.Sequential(
            nn.Linear(8192, 2 * image_embed_size),
            nn.ReLU(),
            nn.Linear(2 * image_embed_size, image_embed_size),
        )
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        shape_length = len(x.shape)
        if shape_length == 2 + 3:
            batch_length = x.shape[0]
            batch_size = x.shape[1]
            x = x.reshape(-1, 3, 128, 128)
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)
        linear_out = self.linear_embed(conv4_out.reshape(x.shape[:-3] + (-1,)))

        if shape_length == 2 + 3:
            linear_out = linear_out.reshape(batch_length, batch_size, -1)
        return linear_out

    def save_checkpoint(self, episode=''):
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
        torch.save(self.state_dict(), self.checkpoint_file + episode)

    def load_checkpoint(self, episode='', best=False):
        if not best:
            self.load_state_dict(torch.load(self.checkpoint_file))  # + episode))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file + episode + '_best'))


class ImageDecodeDCNN(nn.Module):
    def __init__(self, latent_size, chkpt_dir='tmp', name='decoder', device='0'):
        super(ImageDecodeDCNN, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name)

        self.image_embed_model = SimpleMLP(name=name + '_embed', input_dims=(latent_size,),
                                           hidden_dims=64,
                                           output_dims=(256, 4, 4),
                                           layers=3,
                                           chkpt_dir=chkpt_dir,
                                           device=device)

        self.image_deconv_model = DCNN(name=name + '_deconv',
                                       chkpt_dir=chkpt_dir, )

        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x_1 = self.image_embed_model(x)
        batch_shape = x_1.shape[:-3]
        feature_shape = x_1.shape[-3:]
        x_1 = x_1.view((-1,) + feature_shape)
        dec_output = self.image_deconv_model(x_1)
        dec_output = dec_output.view(batch_shape + (3, 64, 64))
        return dec_output

    def save_checkpoint(self, episode):
        torch.save(self.state_dict(), self.checkpoint_file + episode)

    def load_checkpoint(self, episode, best=False):
        if not best:
            self.load_state_dict(torch.load(self.checkpoint_file + episode))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file + '_best'))


class ImageDecodeDCNNHighResol(nn.Module):
    def __init__(self, latent_size, chkpt_dir='tmp', name='decoder', device='0'):
        super(ImageDecodeDCNNHighResol, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name)

        self.image_embed_model = SimpleMLP(name=name + '_embed', input_dims=(latent_size,),
                                           hidden_dims=64,
                                           output_dims=(128, 4, 4),
                                           layers=3,
                                           chkpt_dir=chkpt_dir,
                                           device=device)

        self.image_deconv_model = DCNN(name=name + '_deconv',
                                       chkpt_dir=chkpt_dir, )

        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x_1 = self.image_embed_model(x)
        batch_shape = x_1.shape[:-3]
        feature_shape = x_1.shape[-3:]
        x_1 = x_1.view((-1,) + feature_shape)
        dec_output = self.image_deconv_model(x_1)
        dec_output = dec_output.view(batch_shape + (3, 64, 64))
        return dec_output

    def save_checkpoint(self, episode):
        torch.save(self.state_dict(), self.checkpoint_file + episode)

    def load_checkpoint(self, episode, best=False):
        if not best:
            self.load_state_dict(torch.load(self.checkpoint_file + episode))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file + '_best'))


class DCNN(torch.nn.Module):
    """
    Input image dimensions: [batchSize, Channel, Height, Width]
    Input shape: [64, 128, 4, 4]
    output shape: [64, 3, 64, 64]
    """

    def __init__(self, chkpt_dir='tmp', name='decoder', device='0'):
        super(DCNN, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.deconv_4 = deconv_relu(256, 128, 4, stride=2)
        self.deconv_3 = deconv_relu(128, 64, 4, stride=2)
        self.deconv_2 = deconv_relu(64, 32, 4, stride=2)
        # self.deconv_1 = deconv_sigmoid(16, 3, 4, stride=2)
        self.deconv_1 = deconv_sigmoid(32, 3, 4, stride=2)
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        deconv4_out = self.deconv_4(x)
        deconv3_out = self.deconv_3(deconv4_out)
        deconv2_out = self.deconv_2(deconv3_out)
        predict_out = self.deconv_1(deconv2_out)
        return predict_out

    def save_checkpoint(self):
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, best=False):
        if not best:
            self.load_state_dict(torch.load(self.checkpoint_file))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file + '_best'))


class DCNNHighResol(torch.nn.Module):
    """
    Input image dimensions: [batchSize, Channel, Height, Width]
    Input shape: [64, 128, 4, 4]
    output shape: [64, 3, 64, 64]
    """

    def __init__(self, chkpt_dir='tmp', name='decoder', device='0'):
        super(DCNNHighResol, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.deconv_4 = deconv_relu(128, 64, 4, stride=2)
        self.deconv_3 = deconv_relu(64, 32, 4, stride=2)
        self.deconv_2 = deconv_relu(32, 16, 4, stride=2)
        # self.deconv_1 = deconv_sigmoid(16, 3, 4, stride=2)
        self.deconv_1 = deconv_sigmoid(16, 3, 4, stride=2)
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        deconv4_out = self.deconv_4(x)
        deconv3_out = self.deconv_3(deconv4_out)
        deconv2_out = self.deconv_2(deconv3_out)
        predict_out = self.deconv_1(deconv2_out)
        return predict_out

    def save_checkpoint(self):
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, best=False):
        if not best:
            self.load_state_dict(torch.load(self.checkpoint_file))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file + '_best'))


class NormalMLP(nn.Module):
    def __init__(self, input_dims, hidden_dims,
                 output_dims, layers: int, activation=nn.ELU, min=1e-4, max=10.0, name='normal_mlp',
                 chkpt_dir='tmp', device='0'):
        super().__init__()
        self._output_shape = output_dims
        self._layers = layers
        self._hidden_size = hidden_dims
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self._feature_size = input_dims[0]
        # Defining the structure of the NN
        self.model = self.build_model()
        self.soft_plus = nn.Softplus()

        self._min = min
        self._max = max

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size, 2 * int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, features):
        shape_len = len(features.shape)
        if shape_len == 3:
            batch = features.shape[1]
            length = features.shape[0]
            features = features.reshape(-1, features.shape[-1])
        dist_inputs = self.model(features)
        reshaped_inputs_mean = dist_inputs[..., :np.prod(self._output_shape)]
        reshaped_inputs_std = dist_inputs[..., np.prod(self._output_shape):]

        reshaped_inputs_std = torch.clamp(self.soft_plus(reshaped_inputs_std), min=self._min, max=self._max)

        if shape_len == 3:
            reshaped_inputs_mean = reshaped_inputs_mean.reshape(length, batch, -1)
            reshaped_inputs_std = reshaped_inputs_std.reshape(length, batch, -1)
        return torch.distributions.independent.Independent(
            torch.distributions.Normal(reshaped_inputs_mean, reshaped_inputs_std), len(self._output_shape))

    def save_checkpoint(self, episode):
        torch.save(self.state_dict(), self.checkpoint_file + episode)

    def load_checkpoint(self, episode, best=False):
        if not best:
            self.load_state_dict(torch.load(self.checkpoint_file + episode))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file + '_best'))


class CatMLP(nn.Module):
    def __init__(self, input_dims, hidden_dims,
                 output_dims, layers: int, activation=nn.ELU, min=1e-4, max=10.0, name='cat_mlp',
                 chkpt_dir='tmp', device='0'):
        super().__init__()
        self._output_shape = output_dims
        self._layers = layers
        self._hidden_size = hidden_dims
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self._feature_size = input_dims[0]
        # Defining the structure of the NN
        self.model = self.build_model()
        self.soft_plus = nn.Softplus()

        self._min = min
        self._max = max

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, features):
        dist_inputs = self.model(features)
        reshaped_inputs_mean = dist_inputs[..., :np.prod(self._output_shape)]
        return torch.distributions.independent.Independent(
            torch.distributions.OneHotCategorical(logits=reshaped_inputs_mean), 0)

    def save_checkpoint(self, episode):
        torch.save(self.state_dict(), self.checkpoint_file + episode)

    def load_checkpoint(self, episode, best=False):
        if not best:
            self.load_state_dict(torch.load(self.checkpoint_file + episode))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file + '_best'))


class MultiCatMLP(nn.Module):
    def __init__(self, input_dims, hidden_dims,
                 output_dims, layers: int, num_cat=1, activation=nn.ELU, min=1e-4, max=10.0, name='multi_cat_mlp',
                 chkpt_dir='tmp', device='0'):
        super().__init__()
        self._output_shape = output_dims
        self._layers = layers
        self._hidden_size = hidden_dims
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self._feature_size = input_dims[0]
        # Defining the structure of the NN
        self.model = self.build_model()
        self.soft_plus = nn.Softplus()

        self._min = min
        self._max = max

        self._num_cat = num_cat

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.LayerNorm(self._hidden_size)]
        model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, features):
        dist_inputs = self.model(features)
        reshaped_inputs_mean = dist_inputs[..., :np.prod(self._output_shape)]
        # print(torch.max(reshaped_inputs_mean), torch.min(reshaped_inputs_mean))
        return MultiCategoricalDist(inputs=reshaped_inputs_mean, num_cat=self._num_cat)

    def save_checkpoint(self, episode):
        torch.save(self.state_dict(), self.checkpoint_file + episode)

    def load_checkpoint(self, episode, best=False):
        if not best:
            self.load_state_dict(torch.load(self.checkpoint_file + episode))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file + '_best'))


class SimpleMLPBN(nn.Module):
    def __init__(self, input_dims, hidden_dims,
                 output_dims, layers: int, activation=nn.ELU, name='simple_mlp',
                 chkpt_dir='tmp', device='0'):
        super().__init__()
        self._output_shape = output_dims
        self._layers = layers
        self._hidden_size = hidden_dims
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self._feature_size = input_dims[0]
        # Defining the structure of the NN
        self.model = self.build_model()
        self.soft_plus = nn.Softplus()

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [nn.BatchNorm1d(self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, features):
        if len(features.shape) == 3:
            batch_len = features.shape[0]
            batch_size = features.shape[1]
            feature_size = features.shape[2]
            features = features.view(-1, feature_size)
            outputs = self.model(features)
            reshaped_outputs = torch.reshape(outputs[..., :np.prod(self._output_shape)],
                                             features.shape[:-1] + self._output_shape)
            features = features.view(batch_len, batch_size, feature_size)
            reshaped_outputs = reshaped_outputs.view(batch_len, batch_size, reshaped_outputs.shape[-1])
        else:
            outputs = self.model(features)
            reshaped_outputs = torch.reshape(outputs[..., :np.prod(self._output_shape)],
                                             features.shape[:-1] + self._output_shape)
        # print("\tIn Model: input size", features.size(),
        #       "output size", reshaped_outputs.size())
        return reshaped_outputs

    def save_checkpoint(self, episode):
        torch.save(self.state_dict(), self.checkpoint_file + episode)

    def load_checkpoint(self, episode, best=False):
        if not best:
            self.load_state_dict(torch.load(self.checkpoint_file + episode))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file + '_best'))


class SimpleMLP(nn.Module):
    def __init__(self, input_dims, hidden_dims,
                 output_dims, layers: int, activation=nn.ELU, name='simple_mlp',
                 chkpt_dir='tmp', device='0'):
        super().__init__()
        self._output_shape = output_dims
        self._layers = layers
        self._hidden_size = hidden_dims
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self._feature_size = input_dims[0]
        # Defining the structure of the NN
        self.model = self.build_model()
        self.soft_plus = nn.Softplus()

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, features):
        outputs = self.model(features)
        reshaped_outputs = torch.reshape(outputs[..., :np.prod(self._output_shape)],
                                         features.shape[:-1] + self._output_shape)
        # print("\tIn Model: input size", features.size(),
        #       "output size", reshaped_outputs.size())
        return reshaped_outputs

    def save_checkpoint(self, episode):
        torch.save(self.state_dict(), self.checkpoint_file + episode)

    def load_checkpoint(self, episode, best=False):
        if not best:
            self.load_state_dict(torch.load(self.checkpoint_file + episode))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file + '_best'))


class MemoGRU(nn.Module):
    def __init__(self, input_size, h_size, name='simple_mlp',
                 chkpt_dir='tmp', device='0'):
        super().__init__()
        self.memo_gru = nn.RNN(input_size=input_size, hidden_size=h_size, num_layers=1, bidirectional=True)
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, x, _h):
        x_shape_len = len(x.shape)
        if x_shape_len == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x_shape_len == 2:
            x = x.unsqueeze(0)

        _h_shape_len = len(_h.shape)
        if _h_shape_len == 1:
            _h = _h.unsqueeze(0).unsqueeze(0)
        elif _h_shape_len == 2:
            _h = _h.unsqueeze(0)

        h, _ = self.memo_gru(x, _h)

        if x_shape_len == 1:
            h = h.squeeze(0).squeeze(0)
        elif x_shape_len == 2:
            h = h.squeeze(0)

        return h

    def save_checkpoint(self, episode):
        torch.save(self.state_dict(), self.checkpoint_file + episode)

    def load_checkpoint(self, episode, best=False):
        if not best:
            self.load_state_dict(torch.load(self.checkpoint_file + episode))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file + '_best'))


class MultiModalEmbed(nn.Module):
    def __init__(self, modality_info_dict, embed_size=32, name='multimodal_embed', mask_data=False, device='0'):
        super().__init__()
        #  modality_info_dict = {f'{modality_name}': {'modality_type', 'modality_size', 'embed_size'}}
        self.modality_modules = nn.ModuleDict()
        self.modality_info_dict = modality_info_dict
        self.cat_embed_size = 0
        self.fuse_embed_size = embed_size
        for modality_name, modality_info in modality_info_dict.items():
            if modality_info['modality_type'] == 'image':
                self.modality_modules[modality_name] = ImageEmbedCNN(name=name + modality_name,
                                                                     image_embed_size=modality_info['embed_size'],
                                                                     device=device)
            elif modality_info['modality_type'] == 'state':
                self.modality_modules[modality_name] = SimpleMLP(name=name + modality_name,
                                                                 input_dims=modality_info['modality_size'],
                                                                 hidden_dims=64,
                                                                 output_dims=(modality_info['embed_size'],), layers=2,
                                                                 device=device)
            else:
                self.modality_modules[modality_name] = SimpleMLP(name=name + modality_name,
                                                                 input_dims=modality_info['modality_size'],
                                                                 hidden_dims=64,
                                                                 output_dims=(modality_info['embed_size'],), layers=2,
                                                                 device=device)

            self.cat_embed_size += modality_info['embed_size']

        self.fuse_model = SimpleMLP(name=name + 'fuse_model',
                                    input_dims=(self.cat_embed_size,),
                                    hidden_dims=64,
                                    output_dims=(self.fuse_embed_size,), layers=2,
                                    device=device)

        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        self.mask_data = mask_data

    def forward(self, modality_dict):
        #  modality_info_dict = {f'{modality_name}': {'type': type, 'data': modality_data}}
        modality_embed = []
        for modality_name, modality_data in modality_dict.items():
            if 'mask' in modality_data and self.mask_data:
                mask = modality_data['mask']
                # if len(mask.shape) != len(modality_data['data'].shape):
                #     mask = mask.unsqueeze(-1).unsqueeze(-1)
                modality_embed += [self.modality_modules[modality_name](modality_data['data']) * mask]
            else:
                modality_embed += [self.modality_modules[modality_name](modality_data['data'])]
        modality_embed = torch.cat(modality_embed, dim=-1)
        fuse_embed = self.fuse_model(modality_embed)
        return fuse_embed

    def forward_default(self, ):
        # if no observation
        fuse_embed = torch.zeros((self.fuse_embed_size,)).to(self.device)
        return fuse_embed

    def save_checkpoint(self, episode):
        for modality_name in self.modality_info_dict.keys():
            self.modality_modules[modality_name].save_checkpoint(episode)

        self.fuse_model.save_checkpoint(episode)

    def load_checkpoint(self, episode, best=False):
        if not best:
            for modality_name in self.modality_info_dict.keys():
                self.modality_modules[modality_name].load_checkpoint(episode)

            self.fuse_model.load_checkpoint(episode)
        else:
            for modality_name in self.modality_names:
                self.modality_modules[modality_name].load_checkpoint(episode, best=True)

            self.fuse_model.load_checkpoint(episode, best=True)


class MultiModalDecoder(nn.Module):
    def __init__(self, modality_info_dict, latent_size, name='multimodal_embed', chkpt_dir='tmp', device='0'):
        super().__init__()
        #  modality_info_dict = {f'{modality_name}': {'modality_type', 'modality_size', 'latent_size (e.g. use only h'}}
        self.modality_info_dict = modality_info_dict
        self.modality_modules = nn.ModuleDict()
        self.modality_names = []
        for modality_name, modality_info in modality_info_dict.items():
            if modality_info['modality_type'] == 'image':
                self.modality_modules[modality_name] = ImageDecodeDCNN(name=name + modality_name,
                                                                       latent_size=latent_size,
                                                                       device=device)
            elif modality_info['modality_type'] == 'state':
                self.modality_modules[modality_name] = SimpleMLP(name=name + modality_name,
                                                                 input_dims=(latent_size,),
                                                                 hidden_dims=64,
                                                                 output_dims=modality_info['modality_size'], layers=2,
                                                                 device=device)
            else:
                self.modality_modules[modality_name] = SimpleMLP(name=name + modality_name,
                                                                 input_dims=(latent_size,),
                                                                 hidden_dims=64,
                                                                 output_dims=modality_info['modality_size'], layers=2,
                                                                 device=device)

            self.modality_names += [modality_name]

    def forward(self, latent_state):
        modality_rec = {}
        for modality_name, modality_module in self.modality_modules.items():
            modality_rec[modality_name] = modality_module(latent_state)
        return modality_rec

    def save_checkpoint(self, episode):
        for modality_name in self.modality_names:
            self.modality_modules[modality_name].save_checkpoint(episode)

    def load_checkpoint(self, episode, best=False):
        if not best:
            for modality_name in self.modality_names:
                self.modality_modules[modality_name].load_checkpoint(episode)
        else:
            for modality_name in self.modality_names:
                self.modality_modules[modality_name].load_checkpoint(episode, best=True)

# if __name__ == '__main__':
# x = torch.ones((5, 48, 32))
# dist = MultiCategoricalDist(x, (10,), 5)
# import time
# myNet = SimpleMLP(name='sh_encode',
#                   input_dims=(10000,),
#                   hidden_dims=128,
#                   output_dims=(50,), layers=3,
#                   device='0')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     parallel_net = nn.DataParallel(myNet, device_ids=[0, 1])
# myNet.to(device)

# x = torch.ones((1000, 10000)).to(device)
# start = time.time()
# predictions = parallel_net(x)
# stop = time.time()
# print("The time of the run:", stop - start)
# start = time.time()
# predictions = parallel_net(x)
# stop = time.time()
# print("The time of the run:", stop - start)
# start = time.time()
# predictions = parallel_net(x)
# stop = time.time()
# start = time.time()
# predictions = parallel_net(x)
# stop = time.time()
# print("The time of the run:", stop - start)
# start = time.time()
# predictions = parallel_net(x)
# stop = time.time()
# print("The time of the run:", stop - start)
# start = time.time()
# predictions = parallel_net(x)
# stop = time.time()
# print("The time of the run:", stop - start)
# print("Outside: input size", x.size(),
#       "output_size", predictions.size())

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     parallel_net = nn.DataParallel(myNet, device_ids=[0])
# myNet.to(device)

# start = time.time()
# predictions = parallel_net(x)
# stop = time.time()
# print("The time of the run:", stop - start)
# start = time.time()
# predictions = parallel_net(x)
# stop = time.time()
# print("The time of the run:", stop - start)
# start = time.time()
# predictions = parallel_net(x)
# stop = time.time()
# start = time.time()
# predictions = parallel_net(x)
# stop = time.time()
# print("The time of the run:", stop - start)
# start = time.time()
# predictions = parallel_net(x)
# stop = time.time()
# print("The time of the run:", stop - start)
# start = time.time()
# predictions = parallel_net(x)
# stop = time.time()
# print("The time of the run:", stop - start)
# print("Outside: input size", x.size(),
#       "output_size", predictions.size())

# performs a forward pass
# start = time.time()
# predictions = parallel_net(x)
# stop = time.time()
# print("The time of the run:", stop - start)
# start = time.time()
# predictions = parallel_net(x)
# stop = time.time()
# print("The time of the run:", stop - start)
# predictions = parallel_net(x)
# predictions = parallel_net(x)
#
#
#
# myNet = SimpleMLP(name='sh_encode',
#                   input_dims=(5000,),
#                   hidden_dims=128,
#                   output_dims=(50,), layers=3,
#                   device='0')
# x = torch.ones((2500, 5000)).to(myNet.device)
# # performs a forward pass
# start = time.time()
# predictions = myNet(x)
# predictions = myNet(x)
# predictions = myNet(x)
# predictions = myNet(x)
# stop = time.time()
# print("The time of the run:", stop - start)

# computes a loss function
# loss = loss_function(predictions, labels)
# # averages GPU-losses and performs a backward pass
# loss.mean().backward()
# optimizer.step()
