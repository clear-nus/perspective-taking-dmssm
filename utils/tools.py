import numpy as np
import torch
import cv2


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def update_target_networks(target_q, q, update_tau):
    soft_update_from_to(target=target_q, source=q, tau=update_tau)


def get_parameters(modules):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular mskddatrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones((sz, sz),dtype=torch.bool), diagonal=1)
    # return torch.BoolTensor(torch.triu(torch.ones(sz, sz), diagonal=1) ==1)


def generate_square_backward_mask(sz: int):
    """Generates an upper-triangular mskddatrix of -inf, with zeros on diag."""
    return torch.tril(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    # return torch.BoolTensor(torch.triu(torch.ones(sz, sz), diagonal=1) ==1)


def generate_square_mask(sz: int):
    """Generates an matrix of zeros, with zeros on diag."""
    return torch.triu(torch.zeros((sz, sz),dtype=torch.bool), diagonal=1)


def generate_square_random_mask(sz: int, p=0.5):
    '''
    :param sz: size of transformer input embeddings
    :param p: probability for every token to NOT get paid attention to
    :return:
    '''
    # mask = torch.triu(torch.zeros(sz, sz), diagonal=1)
    mask = torch.BoolTensor(torch.rand(size=[sz, sz]) < p).fill_diagonal_(False)
    # mask = (mask + noise) * float('-inf')
    return mask

def generate_key_padding_mask_from_comms_mask(comms_mask: torch.Tensor, dummy_type='None'):
    """Generates a Key Padding Mask from Comms Mask 
    :param comms_mask: 0 means masked
        shape: (batch_length, batch_size, 1), dtype: int, range: [0,1]
        dummy_type: 'None' no dummy variable added;
                    'always': add dummy variable at the begining of the sequence and always unmask it
                    'depend': add dummy variable at the begining of the sequence and unmask it if anything else is masked
        
    :returns pad_mask: True means masked
        shape: (batch_size, batch_length), dtype: bool
    """
    pad_mask = (comms_mask - 1) * - 1
    pad_mask = pad_mask.squeeze(-1)
    pad_mask = pad_mask.transpose(-1,0).type(torch.bool)
    if dummy_type == 'always':
        unmasked_dummy_variable = torch.zeros((pad_mask.shape[0],1),dtype=torch.bool).to(pad_mask.device)
        pad_mask = torch.cat((unmasked_dummy_variable,pad_mask),dim=1)
    elif dummy_type == 'depend':
        unmasked_dummy_variable = torch.any(~pad_mask, dim=1, keepdim=True)
        pad_mask = torch.cat((unmasked_dummy_variable, pad_mask), dim=1)
    elif dummy_type == 'None':
        pass
    else:
        raise NotImplementedError
    return pad_mask


def kl_divergence_cat(p1, p2):
    kl_loss = (p1 * (torch.log(p1 + 1e-9) - torch.log(p2 + 1e-9))).sum(-1).mean()
    return kl_loss


def kl_divergence_normal(mean1, mean2, std1, std2):
    # mean1, std1: post dist
    # mean2 std2: prior dist
    kl_loss = ((std2 + 1e-9).log() - (std1 + 1e-9).log()
               + (std1.pow(2) + (mean2 - mean1).pow(2))
               / (2 * std2.pow(2) + 1e-9) - 0.5).sum(-1).mean()
    return kl_loss


def image_np2torch(x, num_samples=None, device='cpu'):
    if num_samples is not None:
        return torch.as_tensor(x) \
            .to(device).float() \
            .unsqueeze(0) \
            .repeat(num_samples, 1, 1, 1) \
            .transpose(-1, -3).transpose(-1, -2)
    else:
        if x.shape[-1] == 3:
            return torch.as_tensor(x).to(device).float().transpose(-1, -3).transpose(-1, -2)
        else:
            return torch.as_tensor(x).to(device).float()

def torch2np(data, type='image'):
    if type == 'image':
        return data.detach().to('cpu').numpy().transpose(0,1,3,4,2)
    else:
        return data.detach().to('cpu').numpy()

def state_np2torch(x, num_samples=None, device='cpu'):
    if num_samples is not None:
        return torch.as_tensor(x).to(device).float().unsqueeze(0).repeat(num_samples, 1)
    else:
        return torch.as_tensor(x).to(device).float()


def dict_np2torch(data_dict, num_samples=None, device='cpu'):
    data_dict_torch = {}
    for obs_name, obs_dict in data_dict.items():
        obs_type = obs_dict['type']
        if obs_type == 'image':
            data_torch = image_np2torch(obs_dict['data'], num_samples, device)
        elif obs_type == 'state':
            data_torch = state_np2torch(obs_dict['data'], num_samples, device)
        else:
            data_torch = image_np2torch(obs_dict['data'], num_samples, device)

        if 'mask' in obs_dict.keys():
            # if obs_type == 'image':
            #     mask_torch = image_np2torch(obs_dict['mask'], num_samples, device)
            # else:
            #     mask_torch = state_np2torch(obs_dict['mask'], num_samples, device)
            mask_torch = state_np2torch(obs_dict['mask'], num_samples, device)
            data_dict_torch[obs_name] = {'type': obs_type, 'data': data_torch, 'mask': mask_torch}
        else:
            data_dict_torch[obs_name] = {'type': obs_type, 'data': data_torch}
    return data_dict_torch


def env_np2torch(obs_info, num_samples=None, device='cpu'):
    # obs_info = {'obs_name': {'type':type, 'data':obs_data}}
    obs_info_torch = {}
    for obs_name, obs_dict in obs_info.items():
        if obs_name == 'agent_obs' or obs_name == 'other_comm' or obs_name == 'self_comm' or obs_name == 'query':
            obs_info_torch[obs_name] = dict_np2torch(obs_dict, num_samples, device)
        else:
            obs_type = obs_dict['type']
            if obs_type == 'image':
                obs_torch_data = image_np2torch(obs_dict['data'], num_samples, device)
            elif obs_type == 'state':
                obs_torch_data = state_np2torch(obs_dict['data'], num_samples, device)
            else:
                obs_torch_data = image_np2torch(obs_dict['data'], num_samples, device)

            obs_info_torch[obs_name] = {'type': obs_type, 'data': obs_torch_data}
    return obs_info_torch


def image_torch2np(image, size=(256, 256)):
    # image (batch, channel, height, width)
    image_np = image.transpose(-3, -1).transpose(-3, -2).cpu().detach().numpy()
    image_np_resize = []
    for i in range(image_np.shape[0]):
        image_np_resize += [cv2.resize(image_np[i], dsize=size)]
    image_np_resize = np.stack(image_np_resize, axis=0)
    return image_np_resize


def cat_fuse(means, masks):
    # means = (num_modality, ...), masks = (num_modality, ...)
    if masks is None:
        fuse_mean = means.mean(dim=0)
    else:
        fuse_mean = (means * masks).sum(-1) / masks.sum(0)
    return fuse_mean
