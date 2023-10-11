import torch
import torch.nn as nn
from utils.networks import *
from utils.tools import *
from agents.models.human_policy_model import PolicyModel


class HVAECCat(nn.Module):
    def __init__(self, h_size, s_size, zp_size, xp_size, action_size, num_cat, name,
                 agent_obs_info_dict, agent_comm_info_dict, pi_info_dict,
                 obs_fuse_embed_size=64,
                 transformer_embed_size=64,
                 device='0', reward_mode='st_at', use_transformer=True):
        super().__init__()
        self.name = name
        latent_size = h_size + s_size
        transformer_embed_size = s_size
        multimodal_embed_size = obs_fuse_embed_size
        self.use_transformer = use_transformer
        self.pi_info_dict = pi_info_dict


        self.h_embed_size = 0
        for pi_name, pi_info in pi_info_dict.items():
            self.h_embed_size += pi_info['embed_size']

        self.s_embed_size = 0
        for agent_obs_name, agent_obs_info in agent_obs_info_dict.items():
            self.s_embed_size += agent_obs_info['embed_size']

        self.e_embed_size = 0
        for comm_name, comm_info in agent_comm_info_dict.items():
            self.e_embed_size += comm_info['embed_size']

        self.h_embed_size = multimodal_embed_size
        self.s_embed_size = multimodal_embed_size
        self.e_embed_size = multimodal_embed_size

        # state sizes
        self.latent_size = latent_size
        self.action_size = action_size
        self.h_size = h_size
        self.s_size = s_size
        self.zp_size = zp_size
        self.num_cat = num_cat

        self.reward_mode = reward_mode

        self.device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'

        # self.pt_model = NormalMLP(name=name + 'pt',
        #                           input_dims=(self.h_embed_size + zp_size,),
        #                           hidden_dims=64,
        #                           output_dims=(multimodal_embed_size,), layers=3,
        #                           device=device)
        self.pt_model = PerspTakingNetwork(agent_obs_info_dict=agent_obs_info_dict,
                                           pi_info_dict=pi_info_dict,
                                           embed_size=32,
                                           pos_size=xp_size,
                                           name=name+'pt',
                                           device=device)
        self.pt_embed_model = MultiModalEmbed(modality_info_dict=pi_info_dict,
                                               embed_size=multimodal_embed_size,
                                               name=name + 'pt_embed_model',
                                               device=device)
        self.pt_decode_model = MultiModalDecoder(modality_info_dict=agent_obs_info_dict,
                                                 latent_size=multimodal_embed_size + xp_size,
                                                 name=name + 'pt_decode_model',
                                                 device=device)

        # p distributions ==============================================================================================
        # p(h_t|s_t, h_t-1)
        self.h_decode_model = MultiCatMLP(name=name + 'h_decode',
                                          input_dims=(s_size + h_size,),
                                          hidden_dims=64,
                                          output_dims=(h_size,), layers=2, num_cat=num_cat,
                                          device=device)

        self.h_decode_init_model = MultiCatMLP(name=name + 'h_decode_init',
                                               input_dims=(s_size,),
                                               hidden_dims=64,
                                               output_dims=(h_size,), layers=3, num_cat=num_cat,
                                               device=device)

        # p(s_t|s_t-1, h_t-1, a_t-1)
        self.s_trans_model = NormalMLP(name=name + 's_trans',
                                       input_dims=(latent_size + zp_size + action_size,),
                                       hidden_dims=64,
                                       output_dims=(s_size,), layers=2,
                                       device=device)

        # p(xg_t|s_t, h_t) or p(xg_t|s_t, h_t)
        self.xh_decode_model = MultiModalDecoder(modality_info_dict=pi_info_dict, latent_size=self.h_size + self.s_size,
                                                 name=name + 'xh_decode', device=device)

        self.h_embed_decode_model = SimpleMLP(name=name + 'h_embed_decode',
                                              input_dims=(h_size + s_size,),
                                              hidden_dims=64,
                                              output_dims=(self.h_embed_size,), layers=2,
                                              device=device)

        # p(xs_t|s_t)
        self.xs_decode_model = MultiModalDecoder(modality_info_dict=agent_obs_info_dict, latent_size=self.s_size,
                                                 name=name + 'xs_decode', device=device)

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
            reward_input_size = h_size + s_size + action_size
        else:
            reward_input_size = h_size + s_size

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
        self.zp_decode_model = SimpleMLP(name=name + 'xp_decode',
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

        # q distributions ==============================================================================================
        # q(z_1:T, h_1:T, s_1:T|xs_1:T) = q(z_t|z_t-1, h_t-1, s_t-1, e_t-1, s_t:T, e_t:T)
        # \prod q(s_t|xs_t)q(h_t|xh_t)q(e_t|c_t)
        # q(e_t|c_t)
        self.e_embed_model = MultiModalEmbed(modality_info_dict=agent_comm_info_dict,
                                             embed_size=multimodal_embed_size,
                                             name=name + 'e_embed',
                                             device=device)

        # q(s_t|x_t)
        self.s_embed_model = MultiModalEmbed(modality_info_dict=agent_obs_info_dict,
                                             embed_size=multimodal_embed_size,
                                             name=name + 's_embed',
                                             device=device)

        # extract features from s_t:T
        if self.use_transformer:
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

        # q(h_t|xh_t)
        self.h_embed_model = MultiModalEmbed(modality_info_dict=pi_info_dict,
                                             embed_size=multimodal_embed_size,
                                             name=name + 'h_embed',
                                             device=device)
        self.h_encode_model = MultiCatMLP(name=name + 'h_encode',
                                          input_dims=(multimodal_embed_size,),
                                          hidden_dims=64,
                                          output_dims=(h_size,), layers=2, num_cat=num_cat,
                                          device=device)

        # q(zp_t|xp_t)
        self.zp_encode_model = NormalMLP(name=name + 'zp_encode',
                                         input_dims=(xp_size,),
                                         hidden_dims=16,
                                         output_dims=(zp_size,), layers=2,
                                         device=device)

        # human model (human policy)
        self.human_model = PolicyModel(z_size=self.latent_size,
                                       action_size=action_size,
                                       agent_obs_info_dict=None,
                                       agent_comm_info_dict=None,
                                       name=name + 'human_policy',
                                       device=device)

    def get_state_post(self, obs_dict=None, env_dict=None, obs_embed_dict=None):
        if obs_dict is not None:
            # input shape (batch_length, batch_size, feature)
            # obs_dict = {'xs':xs_modality_dict, 'xh':xh_modality_dict}
            agent_obs_dict = obs_dict['agent_obs']
            other_comm_dict = obs_dict['other_comm']
            pi_obs_dict = obs_dict['pi_obs']
            xp = env_dict['pos']

            zp_post_dist = self.zp_encode_model(xp)  # get zp_1:T

            if self.training:
                zp_sample = zp_post_dist.rsample()
            else:
                zp_sample = zp_post_dist.sample()

            s_embed = self.s_embed_model(agent_obs_dict)

            e_embed = self.e_embed_model(other_comm_dict)
            e_mask = other_comm_dict['text']['mask']

            if self.use_transformer:
                transformer_embed_s = self.transformer_s(s_embed,
                                                         src_mask=None,
                                                         src_padding_mask=None)

                # create dummy variable (in case of all e_embed is masked)
                dummy_state = torch.zeros_like(e_embed[:1]).float().to(e_embed.device)
                key_padding_mask = generate_key_padding_mask_from_comms_mask(e_mask, dummy_type='depend')
                transformer_embed_e = self.transformer_e(torch.cat([dummy_state, e_embed], dim=0),
                                                         src_mask=None,
                                                         src_padding_mask=key_padding_mask)[1:]
                s_post_dist = self.s_encode_model(torch.cat([transformer_embed_s, transformer_embed_e], dim=-1))
            else:
                gru_s, _ = self.gru_s.memo_gru(s_embed)
                gru_e, _ = self.gru_e.memo_gru(e_embed)
                s_post_dist = self.s_encode_model(torch.cat([gru_s, gru_e], dim=-1))
            # s_post_dist = self.s_encode_model(s_embed)

            if pi_obs_dict is not None:
                h_embed = self.h_embed_model(pi_obs_dict)
                h_post_dist = self.h_encode_model(h_embed)

                if self.training:
                    h_samples = h_post_dist.rsample()
                    s_samples = s_post_dist.rsample()
                else:
                    h_samples = h_post_dist.sample()
                    s_samples = s_post_dist.sample()
                s_means = s_post_dist.mean
                s_stds = s_post_dist.stddev

                h_means = h_post_dist.mean
                return {'h_sample': h_samples, 'h_mean': h_means, 'h_embed': h_embed, 'h_dist': h_post_dist,
                        's_sample': s_samples, 's_mean': s_means, 's_std': s_stds,
                        's_embed': s_embed, 'e_embed': e_embed,
                        'zp_sample': zp_sample, 'zp_mean': zp_post_dist.mean, 'zp_std': zp_post_dist.stddev}
            else:
                if self.training:
                    s_samples = s_post_dist.rsample()
                else:
                    s_samples = s_post_dist.sample()

                s_means = s_post_dist.mean
                s_stds = s_post_dist.stddev

                t_length = s_samples.shape[0]

                h_means = [[]] * t_length
                h_samples = [[]] * t_length

                for t in range(t_length):
                    if t == 0:
                        h_post_dist = self.h_decode_init_model(s_samples[t])
                    else:
                        h_post_dist = self.h_decode_model(torch.cat([s_samples[t], h_sample], dim=-1))
                    if self.training:
                        h_sample = h_post_dist.rsample()
                    else:
                        h_sample = h_post_dist.sample()

                    h_means[t] = h_post_dist.mean
                    h_samples[t] = h_sample

                h_means = torch.stack(h_means, dim=0)
                h_samples = torch.stack(h_samples, dim=0)

                return {'h_sample': h_samples, 'h_mean': h_means, 'h_dist': h_post_dist,
                        's_sample': s_samples, 's_mean': s_means, 's_std': s_stds,
                        's_embed': s_embed, 'e_embed': e_embed,
                        'zp_sample': zp_sample, 'zp_mean': zp_post_dist.mean, 'zp_std': zp_post_dist.stddev}
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
                key_padding_mask = generate_key_padding_mask_from_comms_mask(e_mask, dummy_type='depend')
                transformer_embed_e = self.transformer_e(torch.cat([dummy_state, e_embed], dim=0),
                                                         src_mask=None,
                                                         src_padding_mask=key_padding_mask)[1:]
                s_post_dist = self.s_encode_model(torch.cat([transformer_embed_s, transformer_embed_e], dim=-1))
            else:
                gru_s, _ = self.gru_s.memo_gru(s_embed)
                gru_e, _ = self.gru_e.memo_gru(e_embed)
                s_post_dist = self.s_encode_model(torch.cat([gru_s, gru_e], dim=-1))

            # s_post_dist = self.s_encode_model(s_embed)
            if self.training:
                s_samples = s_post_dist.rsample()
            else:
                s_samples = s_post_dist.sample()

            s_means = s_post_dist.mean
            s_stds = s_post_dist.stddev

            t_length = s_embed.shape[0]

            h_means = [[]] * t_length
            h_samples = [[]] * t_length

            for t in range(t_length):
                if t == 0:
                    h_post_dist = self.h_decode_init_model(s_samples[t])
                else:
                    h_post_dist = self.h_decode_model(torch.cat([s_samples[t], h_sample], dim=-1))

                if self.training:
                    h_sample = h_post_dist.rsample()
                else:
                    h_sample = h_post_dist.sample()

                h_means[t] = h_post_dist.mean
                h_samples[t] = h_sample

            h_means = torch.stack(h_means, dim=0)
            h_samples = torch.stack(h_samples, dim=0)

            return {'h_sample': h_samples, 'h_mean': h_means, 'h_dist': h_post_dist,
                    's_sample': s_samples, 's_mean': s_means, 's_std': s_stds}

    def forward_state_prior(self, _prior_dict, action_dict):
        # _prior_dict = {'h','s', 'zp'}
        # action_dict = {'self_action'}
        _h, _s, _zp = _prior_dict['h'], _prior_dict['s'], _prior_dict['zp']
        action = action_dict['action']

        zp_prior_dist = self.zp_trans_model(torch.cat([_zp, action], dim=-1))

        s_prior_dist = self.s_trans_model(torch.cat([_h, _s, _zp, action], dim=-1))

        if self.training:
            s_sample = s_prior_dist.rsample()
            zp_sample = zp_prior_dist.rsample()
        else:
            s_sample = s_prior_dist.sample()
            zp_sample = zp_prior_dist.sample()

        h_prior_dist = self.h_decode_model(torch.cat([s_sample, _h], dim=-1))

        if self.training:
            h_sample = h_prior_dist.rsample()
            s_sample = s_prior_dist.rsample()
        else:
            h_sample = h_prior_dist.sample()
            s_sample = s_prior_dist.sample()

        return {'h_sample': h_sample, 'h_mean': h_prior_dist.mean,
                's_sample': s_sample, 's_mean': s_prior_dist.mean, 's_std': s_prior_dist.stddev,
                'zp_sample': zp_sample, 'zp_mean': zp_prior_dist.mean, 'zp_std': zp_prior_dist.stddev, }

    def perspective_predict(self, pi_dict, pos):
        # given current robot's belief of the world, predict human's mental state
        # state_dict =  {'h_embed','zp', 'e_embed': previous robot communication to human, 'e_mask'}
        other_obs_dist = self.pt_model(pi_dict, pos)
        return other_obs_dist
    # def perspective_predict(self, state_dict):
    #     # given current robot's belief of the world, predict human's mental state
    #     # state_dict =  {'pi', 'xp', 'e_embed': previous robot communication to human, 'e_mask'}
    #     pi, xp = state_dict['pi'], state_dict['xp']
    #     pt_embed = self.pt_embed_model(pi)
    #     pt_embed = torch.cat([pt_embed, xp], dim=-1)
    #     pt_rec = self.pt_decode_model(pt_embed)
    #     pt_rec = {
    #         modality_name: {
    #             'type': self.pt_decode_model.modality_info_dict[modality_name]['modality_type'],
    #             'data': modality_data,
    #             'mask': 1
    #         }
    #         for modality_name, modality_data in pt_rec.items()}
    #
    #     other_s_embed = self.s_embed_model(pt_rec)
    #
    #     obs_embed_dict = {'s_embed': other_s_embed, 'e_embed': state_dict['e_embed'], 'e_mask': state_dict['e_mask']}
    #
    #     other_post_dict = self.get_state_post(obs_dict=None, obs_embed_dict=obs_embed_dict)
    #
    #     return {'h_sample': other_post_dict['h_sample'], 'h_mean': other_post_dict['h_mean'], 's_embed': other_s_embed,
    #             's_sample': other_post_dict['s_sample'], 's_mean': other_post_dict['s_mean'], 's_std': other_post_dict['s_std']}

    def save_checkpoint(self, episode=''):
        # p distributions
        self.h_decode_model.save_checkpoint(episode)
        self.h_decode_init_model.save_checkpoint(episode)
        self.s_trans_model.save_checkpoint(episode)
        self.zp_trans_model.save_checkpoint(episode)
        self.xs_decode_model.save_checkpoint(episode)
        self.xs_embed_decode_model.save_checkpoint(episode)
        self.s_embed_decode_model.save_checkpoint(episode)
        self.h_embed_decode_model.save_checkpoint(episode)
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
        self.h_embed_model.save_checkpoint(episode)
        self.s_encode_model.save_checkpoint(episode)
        self.h_encode_model.save_checkpoint(episode)
        self.zp_encode_model.save_checkpoint(episode)

        # perspective taking model
        self.pt_model.save_checkpoint(episode)
        self.pt_embed_model.save_checkpoint(episode)
        self.pt_decode_model.save_checkpoint(episode)

        # human policy model
        self.human_model.save_checkpoint(episode)

    def load_checkpoint(self, episode=''):
        # p distributions
        self.h_decode_model.load_checkpoint(episode)
        self.h_decode_init_model.load_checkpoint(episode)
        self.s_trans_model.load_checkpoint(episode)
        self.zp_trans_model.load_checkpoint(episode)
        self.xs_decode_model.load_checkpoint(episode)
        self.xs_embed_decode_model.load_checkpoint(episode)
        self.s_embed_decode_model.load_checkpoint(episode)
        self.xh_decode_model.load_checkpoint(episode)
        self.comm_decode_model.load_checkpoint(episode)
        self.h_embed_decode_model.load_checkpoint(episode)
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
        self.h_embed_model.load_checkpoint(episode)
        self.s_encode_model.load_checkpoint(episode)
        self.h_encode_model.load_checkpoint(episode)
        self.zp_encode_model.load_checkpoint(episode)

        # perspective taking model
        self.pt_model.load_checkpoint(episode)
        self.pt_embed_model.load_checkpoint(episode)
        self.pt_decode_model.load_checkpoint(episode)

        # human policy model
        self.human_model.load_checkpoint(episode)
