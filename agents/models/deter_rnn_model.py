import torch
import torch.nn as nn
from utils.networks import *
from utils.tools import *
from agents.models.human_policy_model import PolicyModel


class DeterRNN(nn.Module):
    def __init__(self, s_size, zp_size, xp_size, action_size, name,
                 agent_obs_info_dict, agent_comm_info_dict, pi_info_dict,
                 obs_fuse_embed_size=64,
                 hidden_s_size=200,
                 hidden_zp_size=10,
                 transformer_embed_size=64,
                 device='0', reward_mode='st_at', use_transformer=True):
        super().__init__()
        # latent_size = s_size + h_size
        transformer_embed_size = s_size
        multimodal_embed_size = obs_fuse_embed_size

        # self.h_embed_size = 0
        # for pi_name, pi_info in pi_info_dict.items():
        #     self.h_embed_size += pi_info['embed_size']

        # self.s_embed_size = 0
        # for agent_obs_name, agent_obs_info in agent_obs_info_dict.items():
        #     self.s_embed_size += agent_obs_info['embed_size']
        #
        # self.e_embed_size = 0
        # for comm_name, comm_info in agent_comm_info_dict.items():
        #     self.e_embed_size += comm_info['embed_size']

        # self.h_embed_size = multimodal_embed_size
        self.s_embed_size = multimodal_embed_size
        self.h_embed_size = multimodal_embed_size
        self.e_embed_size = multimodal_embed_size

        # state sizes
        # self.latent_size = latent_size
        self.action_size = action_size
        # self.h_size = h_size
        self.s_size = s_size
        self.zp_size = zp_size
        # self.num_cat = num_cat
        self.hidden_s_size = hidden_s_size
        self.hidden_zp_size = hidden_zp_size

        self.reward_mode = reward_mode
        self.name = name
        self.device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'

        self.pi_info_dict = pi_info_dict

        ###TO BE CONFIRMED###
        self.pt_model = PerspTakingNetwork(agent_obs_info_dict=agent_obs_info_dict,
                                           pi_info_dict=pi_info_dict,
                                           embed_size=32,
                                           pos_size=xp_size,
                                           name=name + 'pt',
                                           device=device)
        # q distributions ==============================================================================================
        # q(z_1:T, h_1:T, s_1:T|xs_1:T) = q(z_t|z_t-1, h_t-1, s_t-1, e_t-1, s_t:T, e_t:T)
        # \prod q(s_t|xs_t)q(h_t|xh_t)q(e_t|c_t)
        # q(e_t|c_t)
        if use_transformer:
            self.e_embed_model = MultiModalEmbed(modality_info_dict=agent_comm_info_dict,
                                                 embed_size=multimodal_embed_size,
                                                 name=name + 'e_embed',
                                                 device=device)
        else:
            self.e_embed_model = MultiModalEmbed(modality_info_dict=agent_comm_info_dict,
                                                 embed_size=multimodal_embed_size,
                                                 name=name + 'e_embed',
                                                 device=device, mask_data=True)

        # q(s_t|x_t, x_GT)
        # INCLUDE PI INFORMATION INTO AGENT OBS. IN PLACE OPERATION
        # agent_obs_input_info_dict = agent_obs_info_dict.copy()
        # agent_obs_input_info_dict.update(pi_info_dict)
        self.s_embed_model = MultiModalEmbed(modality_info_dict=agent_obs_info_dict,
                                             embed_size=multimodal_embed_size,
                                             name=name + 's_embed',
                                             device=device,
                                             mask_data=False)

        # extract features from s_t:T
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer_s = TransformerSimpleNetwork(name=name + 'transformer_s',
                                                          output_dims=(transformer_embed_size,),
                                                          input_dims=(multimodal_embed_size,),
                                                          trans_hidden_dims=transformer_embed_size,
                                                          ff_hidden_dims=transformer_embed_size,
                                                          num_heads=8,
                                                          layers=2,
                                                          device=device)

            self.transformer_e = TransformerSimpleNetwork(name=name + 'transformer_e',
                                                          output_dims=(transformer_embed_size,),
                                                          input_dims=(multimodal_embed_size,),
                                                          trans_hidden_dims=transformer_embed_size,
                                                          ff_hidden_dims=transformer_embed_size,
                                                          num_heads=8,
                                                          layers=2,
                                                          device=device)
        else:
            self.gru_s = MemoGRU(input_size=multimodal_embed_size, h_size=int(transformer_embed_size/2), name=name+'gru_s', device=device)
            self.gru_e = MemoGRU(input_size=multimodal_embed_size, h_size=int(transformer_embed_size/2), name=name+'gru_e', device=device)

        self.s_encode_model = NormalMLP(name=name + 's_encode',
                                        input_dims=(2 * transformer_embed_size,),
                                        hidden_dims=64,
                                        output_dims=(s_size,), layers=2,
                                        device=device)

        self.s_trans_model = NormalMLP(name=name + 's_trans',
                                       input_dims=(s_size + zp_size + action_size,),
                                       hidden_dims=64,
                                       output_dims=(s_size,), layers=2,
                                       device=device)

        # MemoGRU(input_size=self.action_size + self.zp_size, h_size=self.s_size, name=name + 's_trans', device=device)

        # NO NEED FOR XH DECODE, BC X AND GT ARE CONCAT TOGETHER, XS_DECODE WILL DECODE EVERYTHING
        # p(xg_t|s_t, h_t) or p(xg_t|s_t, h_t)
        # self.xh_decode_model = MultiModalDecoder(modality_info_dict=pi_info_dict, latent_size=self.s_size,
        #                                          name=name+'xh_decode', device=device)

        # p(xs_t|s_t)
        self.xs_decode_model = MultiModalDecoder(modality_info_dict=agent_obs_info_dict, latent_size=self.s_size,
                                                 name=name + 'xs_decode', device=device)

        self.xh_embed_decode_model = SimpleMLP(name=name + 'xh_embed_decode',
                                               input_dims=(s_size,),
                                               hidden_dims=64,
                                               output_dims=(self.h_embed_size,), layers=2,
                                               device=device)

        self.xh_decode_model = MultiModalDecoder(modality_info_dict=pi_info_dict, latent_size=self.s_size,
                                                 name=name + 'xh_decode', device=device)

        # for debugging only
        self.xs_embed_decode_model = MultiModalDecoder(modality_info_dict=agent_obs_info_dict, latent_size=self.s_embed_size,
                                                       name=name + 'xs_embed_decode', device=device)

        self.s_embed_decode_model = SimpleMLP(name=name + 's_embed_decode',
                                              input_dims=(s_size,),
                                              hidden_dims=64,
                                              output_dims=(self.s_embed_size,), layers=2,
                                              device=device)

        # for debugging and storing history
        self.comm_decode_model = MultiModalDecoder(modality_info_dict=agent_comm_info_dict, latent_size=self.e_embed_size,
                                                   name=name + 'comm_decode', device=device)

        # p(r_t|h_t, s_t)
        if reward_mode == 'st_at':
            reward_input_size = s_size + action_size
        else:
            reward_input_size = s_size

        # robot reward
        self.r_reward_decode_model = NormalMLP(name=name + 'robot_reward_decode',
                                               input_dims=(reward_input_size,),
                                               hidden_dims=200,
                                               output_dims=(1,), layers=3,
                                               device=device)
        self.r_done_decode_model = CatMLP(name=name + 'robot_done_decode',
                                          input_dims=(reward_input_size,),
                                          hidden_dims=200,
                                          output_dims=(2,), layers=3,
                                          device=device)

        # estimated human reward
        self.h_reward_decode_model = NormalMLP(name=name + 'human_reward_decode',
                                               input_dims=(reward_input_size,),
                                               hidden_dims=200,
                                               output_dims=(1,), layers=3,
                                               device=device)
        self.h_done_decode_model = CatMLP(name=name + 'human_done_decode',
                                          input_dims=(reward_input_size,),
                                          hidden_dims=200,
                                          output_dims=(2,), layers=3,
                                          device=device)

        # p(xp_t|zp_t)
        self.zp_decode_model = NormalMLP(name=name + 'xp_decode',
                                         input_dims=(zp_size,),
                                         hidden_dims=16,
                                         output_dims=(xp_size,), layers=2,
                                         device=device)

        # p(zp_t+1|zp_t, ar_t)
        self.zp_trans_model = NormalMLP(name=name + 'zp_trans',
                                        input_dims=(zp_size + action_size,),
                                        hidden_dims=16,
                                        output_dims=(zp_size,), layers=2,
                                        device=device)

        # q(zp_t|xp_t)
        self.zp_encode_model = NormalMLP(name=name + 'zp_encode',
                                         input_dims=(xp_size,),
                                         hidden_dims=16,
                                         output_dims=(zp_size,), layers=2,
                                         device=device)

        # human model (human policy)
        self.human_model = PolicyModel(z_size=self.s_size,
                                       action_size=action_size,
                                       agent_obs_info_dict=None,
                                       agent_comm_info_dict=None,
                                       name=name + 'human_policy')

    def get_state_post(self, obs_dict=None, env_dict=None, obs_embed_dict=None):
        if obs_dict is not None:
            # input shape (batch_length, batch_size, feature)
            # obs_dict = {'xs':xs_modality_dict, 'xh':xh_modality_dict}
            agent_obs_dict = obs_dict['agent_obs']
            other_comm_dict = obs_dict['other_comm']
            xp = env_dict['pos']

            # include x_GT into x to obtain s_embed

            zp = self.zp_encode_model(xp).mean  # get zp_1:T

            s_embed = self.s_embed_model(agent_obs_dict)

            e_embed = self.e_embed_model(other_comm_dict)
            e_mask = other_comm_dict['text']['mask']

            if self.use_transformer:
                transformer_embed_s = self.transformer_s(s_embed,
                                                         src_mask=None,
                                                         src_padding_mask=None)

                # create dummy variable (in case of all e_embed is masked)
                dummy_state = torch.zeros_like(e_embed[:1]).float().to(e_embed.device)
                key_padding_mask = generate_key_padding_mask_from_comms_mask(e_mask, dummy_type='always')
                transformer_embed_e = self.transformer_e(torch.cat([dummy_state, e_embed], dim=0),
                                                         src_mask=None,
                                                         src_padding_mask=key_padding_mask)[1:]
                s = self.s_encode_model(torch.cat([transformer_embed_s, transformer_embed_e], dim=-1)).mean
            else:
                gru_s, _ = self.gru_s.memo_gru(s_embed)
                gru_e, _ = self.gru_e.memo_gru(e_embed)
                s = self.s_encode_model(torch.cat([gru_s, gru_e], dim=-1)).mean

            h_embed = self.xh_embed_decode_model(s)

            return {'h_sample': None, 'h_mean': None, 'h_embed': h_embed, 'h_dist': None,
                    's_sample': s, 's_mean': s, 's_std': None,
                    's_embed': s_embed, 'e_embed': e_embed,
                    'zp_sample': zp, 'zp_mean': zp, 'zp_std': None}
        else:
            # obs_embed_dict = {'s': robot s, 'e': e from human comm}
            s_embed = obs_embed_dict['s_embed']
            e_embed = obs_embed_dict['e_embed']
            e_mask = obs_embed_dict['e_mask']
            if self.use_transformer:
                transformer_embed_s = self.transformer_s(s_embed,
                                                         src_mask=None,
                                                         src_padding_mask=None)

                # create dummy variable (in case of all e_embed is masked)
                dummy_state = torch.zeros_like(e_embed[:1]).float().to(e_embed.device)
                key_padding_mask = generate_key_padding_mask_from_comms_mask(e_mask, dummy_type='always')
                transformer_embed_e = self.transformer_e(torch.cat([dummy_state, e_embed], dim=0),
                                                         src_mask=None,
                                                         src_padding_mask=key_padding_mask)[1:]
                s = self.s_encode_model(torch.cat([transformer_embed_s, transformer_embed_e], dim=-1)).mean
            else:
                gru_s, _ = self.gru_s.memo_gru(s_embed)
                gru_e, _ = self.gru_e.memo_gru(e_embed)
                s = self.s_encode_model(torch.cat([gru_s, gru_e], dim=-1)).mean

            h_embed = self.xh_embed_decode_model(s)
            return {'h_sample': None, 'h_mean': None, 'h_dist': None, 'h_embed': h_embed,
                    's_sample': s, 's_mean': s, 's_std': None}

    def forward_state_prior(self, _prior_dict, action_dict):
        # _prior_dict = {'h','s', 'zp'}
        # action_dict = {'self_action'}
        _s, _zp, = _prior_dict['s'], _prior_dict['zp']
        action = action_dict['action']

        zp = self.zp_trans_model(torch.cat([_zp, action], dim=-1)).mean

        s = self.s_trans_model(torch.cat([_s, _zp, action], dim=-1)).mean

        return {'h_sample': None, 'h_mean': None,
                's_sample': s, 's_mean': s, 's_std': None,
                'zp_sample': zp, 'zp_mean': zp, 'zp_std': None}

    def perspective_predict(self, pi_dict, pos):
        # given current robot's belief of the world, predict human's mental state
        # state_dict =  {'h_embed','zp', 'e_embed': previous robot communication to human, 'e_mask'}
        other_obs_dist = self.pt_model(pi_dict, pos)
        return other_obs_dist

    def get_default_pi(self, batch_shape=None):
        default_pi = {}
        for pi_name, pi_info in self.pi_info_dict.items():
            pi_data = torch.zeros(pi_info['modality_size']).float().to(self.device)
            pi_mask = torch.zeros((1,)).float().to(self.device)
            modality_type = pi_info['modality_type']
            if batch_shape is not None:
                if modality_type == 'image':
                    pi_data = pi_data.repeat(batch_shape + (1, 1, 1))
                elif modality_type == 'state':
                    pi_data = pi_data.repeat(batch_shape + (1,))
                pi_mask = pi_mask.repeat(batch_shape + (1,))
            default_pi[pi_name] = {'type': modality_type, 'data': pi_data, 'mask': pi_mask}
        return default_pi

    def save_checkpoint(self, episode=''):
        # p distributions
        # self.h_decode_model.save_checkpoint(episode)
        # self.h_decode_init_model.save_checkpoint(episode)
        # self.s_trans_model.save_checkpoint(episode)
        checkpoint_file = os.path.join(self.zp_trans_model.checkpoint_dir, self.name + 's_trans')
        torch.save(self.s_trans_model.state_dict(), checkpoint_file + episode)

        self.zp_trans_model.save_checkpoint(episode)
        self.xs_decode_model.save_checkpoint(episode)
        self.xs_embed_decode_model.save_checkpoint(episode)
        self.s_embed_decode_model.save_checkpoint(episode)
        self.xh_embed_decode_model.save_checkpoint(episode)
        self.xh_decode_model.save_checkpoint(episode)
        self.comm_decode_model.save_checkpoint(episode)
        self.zp_decode_model.save_checkpoint(episode)

        self.r_reward_decode_model.save_checkpoint(episode)
        self.r_done_decode_model.save_checkpoint(episode)
        self.h_reward_decode_model.save_checkpoint(episode)
        self.h_done_decode_model.save_checkpoint(episode)

        # q distributions
        if self.use_transformer:
            self.transformer_s.save_checkpoint(episode)
            self.transformer_e.save_checkpoint(episode)
        else:
            self.gru_s.save_checkpoint(episode)
            self.gru_e.save_checkpoint(episode)

        self.s_embed_model.save_checkpoint(episode)
        self.e_embed_model.save_checkpoint(episode)
        # self.h_embed_model.save_checkpoint(episode)
        self.s_encode_model.save_checkpoint(episode)
        # self.h_encode_model.save_checkpoint(episode)
        self.zp_encode_model.save_checkpoint(episode)

        # perspective taking model
        self.pt_model.save_checkpoint(episode)

        # human policy model
        self.human_model.save_checkpoint(episode)

    def load_checkpoint(self, episode=''):
        # p distributions
        # self.h_decode_model.load_checkpoint(episode)
        # self.h_decode_init_model.load_checkpoint(episode)
        # self.s_trans_model.load_checkpoint(episode)
        checkpoint_file = os.path.join(self.zp_trans_model.checkpoint_dir, self.name + 's_trans')
        self.s_trans_model.load_state_dict(torch.load(checkpoint_file + episode))
        self.zp_trans_model.load_checkpoint(episode)
        self.xs_decode_model.load_checkpoint(episode)
        self.xs_embed_decode_model.load_checkpoint(episode)
        self.s_embed_decode_model.load_checkpoint(episode)
        self.xh_decode_model.load_checkpoint(episode)
        self.comm_decode_model.load_checkpoint(episode)
        self.xh_embed_decode_model.load_checkpoint(episode)
        self.zp_decode_model.load_checkpoint(episode)

        self.r_reward_decode_model.load_checkpoint(episode)
        self.r_done_decode_model.load_checkpoint(episode)
        self.h_reward_decode_model.load_checkpoint(episode)
        self.h_done_decode_model.load_checkpoint(episode)

        # q distributions
        if self.use_transformer:
            self.transformer_s.load_checkpoint(episode)
            self.transformer_e.load_checkpoint(episode)
        else:
            self.gru_s.load_checkpoint(episode)
            self.gru_e.load_checkpoint(episode)
        self.s_embed_model.load_checkpoint(episode)
        self.e_embed_model.load_checkpoint(episode)
        # self.h_embed_model.load_checkpoint(episode)
        self.s_encode_model.load_checkpoint(episode)
        # self.h_encode_model.load_checkpoint(episode)
        self.zp_encode_model.load_checkpoint(episode)

        # perspective taking model
        self.pt_model.load_checkpoint(episode)

        # human policy model
        self.human_model.load_checkpoint(episode)
