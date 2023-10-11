import os
import numpy as np
import torch
from POMCP.pomcp import POMCP

from utils.networks import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from agents.models.hvae_ccat_model import HVAECCat
from utils.tools import *
import cv2


def get_belief_key(belief):
    # belief: {'self_z': distribution of z_t-1, 'self_zp': distribution of robot zp_t-1, 'other_zp'}
    belief_key = torch.cat([belief['self_z'], belief['self_zp'], belief['other_zp']], dim=-1).cpu().detach().numpy()
    belief_key = f'{belief_key}'
    return belief_key


class HVAECCatPlanAgent(nn.Module):
    def __init__(self,
                 model: HVAECCat,
                 comm_generator,
                 feedback_generator,
                 query_generator,
                 env,
                 inference_length
                 ):
        super().__init__()
        # models
        self.model = model
        self.comm_generator = comm_generator
        self.feedback_generator = feedback_generator
        self.query_generator = query_generator

        # env
        self.env = env

        # # action space
        # self.comm_actions = comm_generator.get_actions(model.device)
        # self.phy_actions = env.get_phy_actions(model.device)
        # self.query_actions = query_generator.get_actions(model.device)

        # robot's belief of the world (time, sample_size, feature)
        self.self_s = None
        self.self_h = None
        self.self_s_embed = None
        self.self_e_embed = None
        self.self_e_mask = None
        self.self_h_dist = None
        self.self_zp = None

        # estimation of human's belief of the world (time, sample_size, feature)
        self.other_s = None
        self.other_h = None
        self.other_s_embed = None
        self.other_e_embed = None
        self.other_e_mask = None
        self.other_zp = None
        self.other_pos = None

        self.inference_length = inference_length

        self.reward_mode = self.model.reward_mode

        self.device = model.device
        self.to(self.device)

        self.action_size = env.action_size

        self.num_h_human = 75

        if not os.path.exists('runs'):
            os.makedirs('runs')
        self.writer = SummaryWriter(log_dir=f'runs/plan/{self.model.name}')
        self.writer_step = 0

    def get_init_dict(self):
        # env_history: {'self_s': a sequence of s_1:t-1, 'self_e', 'other_e', 'self_zp', 'other_zp'}
        # belief: {'self_z': distribution of z_t-1}
        belief = {'self_s': self.self_s[-1, 0], 'self_zp': self.self_zp[-1, 0], 'other_zp': self.other_zp[-1, 0]}

        # for i in range(10):
        #     self_s = self.self_s[-1]
        #     # self_h = self.model.h_decode_init_model_s(self_s).sample()
        #     self_h = self.self_h_dist.sample()
        #     robot_pi = self.model.xh_decode_model(torch.cat([self_h, self_s], dim=-1))['text']
        #     # print(robot_pi)
        #     robot_pi = self.model.xh_decode_model(torch.cat([self.self_h_dist.sample()[0], self.self_s[-1, 0]], dim=-1))['text']
        #     print(robot_pi)
        # print('')

        history = {'self_s_embed': self.self_s_embed[:, 0],
                   'self_e_embed': self.self_e_embed[:, 0],
                   'other_e_embed': self.other_e_embed[:, 0],
                   'self_e_mask': self.self_e_mask[:, 0],
                   'other_e_mask': self.other_e_mask[:, 0],
                   'self_zp': self.self_zp[:, 0],
                   'other_zp': self.other_zp[:, 0]}

        init_dict = {'belief': belief,
                     'history': history}

        return init_dict

    def reset(self):
        # robot's belief of the world (time, sample_size, feature)
        self.self_s = None
        self.self_h = None
        self.self_s_embed = None
        self.self_e_embed = None
        self.self_e_mask = None
        self.self_h_dist = None
        self.self_zp = None

        # estimation of human's belief of the world (time, sample_size, feature)
        self.other_s = None
        self.other_h = None
        self.other_s_embed = None
        self.other_e_embed = None
        self.other_e_mask = None
        self.other_zp = None
        self.other_pos = None

    def forward_state_prior(self, belief, history, action_dict):
        # action_dict = {'robot': robot_action, 'human': human_action, 'robot_query': robot_question (last time step)}
        # e.g. robot_action: {'action': action, 'type': comm or phy}
        # history: {'self_s': a sequence of s_1:t-1, 'self_e', 'other_e', 'self_zp', 'other_zp', 'self_e_mask', 'other_e_mask'}
        # belief: {'self_h_dist': distribution of z_t-1, 'self_s':, 'self_zp': zp_t-1, 'other_zp'}

        if action_dict['agent'] == 'robot':  # robot takes action
            # sample next environment state
            # sample z_t-1, s_t-1 and h_t-1
            # _self_h_state = belief['self_h_dist'].sample()
            obs_embed_dict = {'s_embed': history['self_s_embed'].unsqueeze(1),
                              'e_embed': history['other_e_embed'].unsqueeze(1),
                              'e_mask': history['other_e_mask'].unsqueeze(1)}
            post_dict = self.model.get_state_post(obs_dict=None, env_dict=None, obs_embed_dict=obs_embed_dict)
            _self_h_state = post_dict['h_dist'].sample()[0]
            _self_s_state = belief['self_s']

            if action_dict['type'] == 'phy':
                # p(z_t|z_t-1, s_t-1, h_t-1, a_t-1)
                self_s_prior_dist_ = self.model.s_trans_model(torch.cat([_self_h_state, _self_s_state,
                                                                         belief['self_zp'], action_dict['data']], dim=-1))
                self_s_state_ = self_s_prior_dist_.sample()
                self_s_embed_ = self.model.s_embed_decode_model(self_s_state_)

                self_zp_prior_dist_ = self.model.zp_trans_model(torch.cat([belief['self_zp'], action_dict['data']], dim=-1))

                self_zp_state_ = self_zp_prior_dist_.mean
                other_zp_state_ = belief['other_zp']

                self_e_embed_ = self.get_default_e_embed()
                other_e_embed_ = self.get_default_e_embed()

                # if robot communicate
                self_e_mask = torch.tensor([0]).to(self_e_embed_.device)
                # if human communicate
                other_e_mask = torch.tensor([0]).to(self_e_embed_.device)

                # obs_rec = {}
                # for obs_name, _ in self.model.xs_embed_decode_model.modality_info_dict.items():
                #     obs_rec[obs_name] = {'type': 'state', 'data': torch.round(self.model.xs_embed_decode_model.modality_modules[obs_name](self_s_embed_)).float()}
                # self_s_embed_ = self.model.s_embed_model(obs_rec)
                #
                # robot_pi = self.model.xh_decode_model(torch.cat([post_dict['h_dist'].sample((5,))[:, 0], self_s_state_.repeat(5, 1)], dim=-1))['text']

                # print(f't-1:{robot_pi}')
            # update communication history
            elif action_dict['type'] == 'comm':
                # if robot's action is to communicate
                self_s_embed_ = history['self_s_embed'][-1]

                self_zp_state_ = belief['self_zp']

                other_zp_state_ = belief['other_zp']
                # if robot's action is to communicate
                self_e_embed_ = self.comm_generator.generate_text(self_s_embed_, action_dict['data'])

                other_e_embed_ = self.get_default_e_embed()

                # if robot communicate
                self_e_mask = torch.tensor([1]).to(self_e_embed_.device)
                # if human communicate
                other_e_mask = torch.tensor([0]).to(self_e_embed_.device)
            elif action_dict['type'] == 'query':
                # if robot's action is to ask question
                # self_h_prior_dist_ = belief['self_h_dist']
                # self_h_state_ = self_h_prior_dist_.sample()
                # obs_embed_dict = {'s_embed': history['self_s_embed'].unsqueeze(1),
                #                   'e_embed': history['other_e_embed'].unsqueeze(1),
                #                   'e_mask': history['other_e_mask'].unsqueeze(1)}
                # post_dict = self.model.get_state_post(obs_dict=None, env_dict=None, obs_embed_dict=obs_embed_dict)
                # self_h_state_ = post_dict['h_dist'].sample()

                # self_s_state_ = belief['self_s']
                self_h_embed_ = self.model.h_embed_decode_model(torch.cat([_self_h_state, _self_s_state], dim=-1))
                self_s_embed_ = history['self_s_embed'][-1]
                self_zp_state_ = belief['self_zp']

                other_zp_state_ = belief['other_zp']

                self_e_embed_ = self.get_default_e_embed()

                # generate query
                query = self.query_generator.generate_text(other_zp_state_, action_dict['data'])
                # estimate human's s
                _other_s_embed = self.model.pt_model(torch.cat([self_h_embed_, other_zp_state_], dim=-1)).mean
                other_e_embed_ = self.feedback_generator.generate_text(s_state=_other_s_embed,
                                                                       query=query)
                # comm_rec = {}
                # for comm_name, _ in self.model.comm_decode_model.modality_info_dict.items():
                #     comm_rec[comm_name] = {'type': 'state', 'data': torch.round(self.model.comm_decode_model.modality_modules[comm_name](other_e_embed_)).float()}
                # other_e_embed_ = self.model.e_embed_model(comm_rec)
                #
                # robot_pi1 = self.model.xh_decode_model(torch.cat([self_h_state_, self_s_state_], dim=-1))['text']
                # robot_pi = self.model.xh_decode_model(torch.cat([post_dict['h_dist'].sample().sample((5,))[:, 0], self_s_state_.repeat(5, 1)], dim=-1))['text']
                #
                # print(f't-1:{robot_pi}, {robot_pi1}, {comm_rec}')
                # if robot communicate
                self_e_mask = torch.tensor([0]).to(self_e_embed_.device)
                # if human communicate
                other_e_mask = torch.tensor([1]).to(self_e_embed_.device)

                # xs_rec = self.model.xs_decode_model(_other_s_embed)['img']
                # # xs_rec = xs_rec.detach().cpu().numpy().transpose(-1, 0).transpose(1, 2)
                # xp_rec = self.model.zp_decode_model(other_zp_state_.repeat(1, 1)).mean[0]
                # comm_rec = self.model.comm_decode_model(other_e_embed_)
            else:
                raise NotImplementedError
        elif action_dict['agent'] == 'human':
            obs_embed_dict = {'s_embed': history['self_s_embed'].unsqueeze(1),
                              'e_embed': history['other_e_embed'].unsqueeze(1),
                              'e_mask': history['other_e_mask'].unsqueeze(1)}
            robot_post_dict = self.model.get_state_post(obs_dict=None, env_dict=None, obs_embed_dict=obs_embed_dict)
            _self_h_states = robot_post_dict['h_sample']
            _self_s_states = robot_post_dict['s_mean']
            _self_h_embed = self.model.h_embed_decode_model(torch.cat([_self_h_states, _self_s_states], dim=-1))
            _other_s_embed = self.model.pt_model(torch.cat([_self_h_embed, history['other_zp'].unsqueeze(1)], dim=-1)).mean

            obs_embed_dict = {'s_embed': _other_s_embed,
                              'e_embed': history['self_e_embed'].unsqueeze(1),
                              'e_mask': history['self_e_mask'].unsqueeze(1)}

            human_post_dict = self.model.get_state_post(obs_dict=None, env_dict=None, obs_embed_dict=obs_embed_dict)
            rl_feature = torch.cat([human_post_dict['h_mean'][-1], human_post_dict['s_mean'][-1]], dim=-1)
            human_action = self.model.human_model.act(z=rl_feature, policy_type='deter')
            human_action[..., :] = 0.0
            human_action[..., 0] = 1.0

            if action_dict['type'] != 'phy':
                raise NotImplementedError

            other_s_ = self.model.s_trans_model(torch.cat([_self_h_states[-1],
                                                           human_post_dict['s_mean'][-1],
                                                           belief['other_zp'].unsqueeze(0),
                                                           human_action], dim=-1)).mean
            other_h_ = self.model.h_decode_model(torch.cat([other_s_, _self_h_states[-1]], dim=-1)).mean
            other_h_embed_ = self.model.h_embed_decode_model(torch.cat([other_h_, other_s_], dim=-1))

            self_s_embed_ = self.model.pt_model(torch.cat([other_h_embed_[0], belief['self_zp']], dim=-1)).mean
            self_zp_state_ = belief['self_zp']

            other_zp_prior_dist_ = self.model.zp_trans_model(
                torch.cat([belief['other_zp'], human_action[0]], dim=-1))

            other_zp_state_ = other_zp_prior_dist_.mean

            # get default None
            self_e_embed_ = self.get_default_e_embed()
            other_e_embed_ = self.get_default_e_embed()

            # if robot communicate
            self_e_mask = torch.tensor([0]).to(self_e_embed_.device)
            # if human communicate
            other_e_mask = torch.tensor([0]).to(self_e_embed_.device)
        else:
            raise NotImplementedError

        # history: {'self_s': a sequence of s_1:t-1, 'self_comm', 'other_comm', 'self_zp', 'other_zp'}
        history_ = {'self_s_embed': torch.cat([history['self_s_embed'], self_s_embed_.unsqueeze(0)], dim=0)[-self.inference_length:],
                    'self_e_embed': torch.cat([history['self_e_embed'], self_e_embed_.unsqueeze(0)], dim=0)[-self.inference_length:],
                    'other_e_embed': torch.cat([history['other_e_embed'], other_e_embed_.unsqueeze(0)], dim=0)[-self.inference_length:],
                    'self_e_mask': torch.cat([history['self_e_mask'], self_e_mask.unsqueeze(0)], dim=0)[-self.inference_length:],
                    'other_e_mask': torch.cat([history['other_e_mask'], other_e_mask.unsqueeze(0)], dim=0)[-self.inference_length:],
                    'self_zp': torch.cat([history['self_zp'], self_zp_state_.unsqueeze(0)], dim=0)[-self.inference_length:],
                    'other_zp': torch.cat([history['other_zp'], other_zp_state_.unsqueeze(0)], dim=0)[-self.inference_length:]}

        # update robot belief
        obs_embed_dict = {'s_embed': history_['self_s_embed'].unsqueeze(1),
                          'e_embed': history_['other_e_embed'].unsqueeze(1),
                          'e_mask': history_['other_e_mask'].unsqueeze(1)}
        self_post_dict = self.model.get_state_post(obs_dict=None, env_dict=None, obs_embed_dict=obs_embed_dict)

        # belief_: {'self_z': distribution of z_t}
        # self_h_dist = MultiCategoricalDist(self_post_dict['h_dist'].mean[0],
        #                                    num_cat=self_post_dict['h_dist'].num_cat, if_prob=True)
        # robot_pi = self.model.xh_decode_model(torch.cat([self_h_dist.sample((5,)), self_post_dict['s_sample'][-1].repeat(5, 1)], dim=-1))['text']
        # print(f't:{robot_pi}')
        # print('')

        belief_ = {'self_s': self_post_dict['s_mean'][-1, 0],
                   'self_zp': self_zp_state_, 'other_zp': other_zp_state_}

        return history_, belief_

    def forward_other_state_post(self, self_post_dict):
        self_h = self_post_dict['h_sample'][-self.inference_length:]
        self_s = self_post_dict['s_sample'][-self.inference_length:]
        other_pos = self.other_pos[-self.inference_length:]
        pi_dict = self.model.xh_decode_model(torch.cat([self_h, self_s], dim=-1))

        for obs_name in self.model.pi_info_dict:
            pi_dict[obs_name] = {'modality_type': self.model.pi_info_dict[obs_name]['modality_type'], 'data': pi_dict[obs_name]}
        other_obs_dict = self.model.perspective_predict(pi_dict, other_pos)
        return other_obs_dict

    def forward_state_post(self, obs_dict, env_dict):
        # obs_dict = {'agent_obs':agent_obs_modality_dict, 'self_comm': robot comm, 'other_comm': human comm}
        # env_dict = {'robot_pos': robot pose, 'human_pos': human pose}
        # (sample_size, feature)

        # update robot states ==========================================================================================
        self_zp_post_dist = self.model.zp_encode_model(env_dict['robot_pos']['data'])
        self.self_zp = self_zp_post_dist.mean.unsqueeze(0) if self.self_zp is None else torch.cat([self.self_zp,
                                                                                                   self_zp_post_dist.mean.unsqueeze(0)], dim=0)

        self_s_embed = self.model.s_embed_model(obs_dict['agent_obs'])
        self.self_s_embed = self_s_embed.unsqueeze(0) if self.self_s_embed is None else torch.cat([self.self_s_embed,
                                                                                                   self_s_embed.unsqueeze(0)], dim=0)

        other_e_embed = self.model.e_embed_model(obs_dict['other_comm'])
        other_e_mask = obs_dict['other_comm']['text']['mask']
        self.other_e_embed = other_e_embed.unsqueeze(0) if self.other_e_embed is None else torch.cat([self.other_e_embed,
                                                                                                      other_e_embed.unsqueeze(0)], dim=0)
        self.other_e_mask = other_e_mask.unsqueeze(0) if self.other_e_mask is None else torch.cat([self.other_e_mask,
                                                                                                   other_e_mask.unsqueeze(0)], dim=0)

        obs_embed_dict = {'s_embed': self.self_s_embed, 'e_embed': self.other_e_embed, 'e_mask': self.other_e_mask}
        self_post_dict = self.model.get_state_post(obs_dict=None, env_dict=None, obs_embed_dict=obs_embed_dict)

        self.self_h_dist = self_post_dict['h_dist']
        self.self_s = self_post_dict['s_mean']
        self.self_h = self_post_dict['h_sample']

        self.self_s = self.self_s[-self.inference_length:]
        self.self_h = self.self_h[-self.inference_length:]
        self.self_s_embed = self.self_s_embed[-self.inference_length:]
        self.other_e_embed = self.other_e_embed[-self.inference_length:]
        self.other_e_mask = self.other_e_mask[-self.inference_length:]
        self.self_zp = self.self_zp[-self.inference_length:]

        if env_dict['human_pos'] is not None:
            # get what robot communicate to human
            self_e_embed = self.model.e_embed_model(obs_dict['self_comm'])
            self_e_mask = obs_dict['self_comm']['text']['mask']
            self.self_e_embed = self_e_embed.unsqueeze(0) if self.self_e_embed is None else torch.cat([self.self_e_embed,
                                                                                                       self_e_embed.unsqueeze(0)], dim=0)
            self.self_e_mask = self_e_mask.unsqueeze(0) if self.self_e_mask is None else torch.cat([self.self_e_mask,
                                                                                                    self_e_mask.unsqueeze(0)], dim=0)

            self.self_e_embed = self.self_e_embed[-self.inference_length:]
            self.self_e_mask = self.self_e_mask[-self.inference_length:]

            other_zp_post_dist = self.model.zp_encode_model(env_dict['human_pos']['data'])
            self.other_zp = other_zp_post_dist.mean.unsqueeze(0) if self.other_zp is None else torch.cat([self.other_zp,
                                                                                                          other_zp_post_dist.mean.unsqueeze(0)],
                                                                                                         dim=0)
            self.other_zp = self.other_zp[-self.inference_length:]

            self.other_pos = env_dict['human_pos']['data'].unsqueeze(0) if self.other_pos is None else torch.cat([self.other_pos,
                                                                                                                  env_dict['human_pos']['data'].unsqueeze(0)],
                                                                                                                 dim=0)
            self.other_pos = self.other_pos[-self.inference_length:]
            # update human states
            # ==========================================================================================
            other_obs_dict = self.forward_other_state_post(self_post_dict)

            # cv2.imshow('Human Obs PT', np.hstack(other_obs_dict['img'][0].cpu().detach().numpy().transpose(0,2,3,1)))
            # # cv2.imshow('human obs Actual', other_obs_dict['img'])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            for key, val in other_obs_dict.items():
                other_obs_dict[key] = {'data': val}
            other_obs_dict['pos']['data'] = self.other_pos
            # take latest s_embed ONLY, and save to buffer!
            self.other_s_embed = self.model.s_embed_model(other_obs_dict)[-self.inference_length:]

            obs_embed_dict = {'s_embed': self.other_s_embed, 'e_embed': self.self_e_embed, 'e_mask': self.self_e_mask}
            other_post_dict = self.model.get_state_post(obs_dict=None, env_dict=None, obs_embed_dict=obs_embed_dict)

            self.other_s = other_post_dict['s_mean']
            self.other_h = other_post_dict['h_sample']

            self.other_s = self.other_s[-self.inference_length:]
            self.other_h = self.other_h[-self.inference_length:]

            # other_s = self.other_s[-1]
            # human_obs = self.model.xs_decode_model(other_s)['img']
            # cv2.imshow('human obs PT other_s', np.hstack(human_obs.cpu().detach().numpy().transpose(0,2,3,1)))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


    def human_policy(self, history):
        # todo fix in the future
        return self.env.human_actions[0]  # do nothing

    def human_agent_act(self):
        # for training human policy during self play
        rl_feature = torch.cat([self.self_h_dist.mean, self.self_s[-1]], dim=-1)
        action = self.model.human_model.act(z=rl_feature, policy_type='deter')
        return action

    def get_default_e_embed(self):
        e_embed = self.model.e_embed_model.forward_default()
        return e_embed

    def get_reward(self, history, belief, action_dict, agent_name):
        # belief: {'self_h_dist': distribution of h_t, 'self_s', 'self_zp', 'other_zp'}
        samle_size = 20
        # self_h_samples = belief['self_h_dist'].sample((samle_size,))
        obs_embed_dict = {'s_embed': history['self_s_embed'].unsqueeze(1).repeat(1, samle_size, 1),
                          'e_embed': history['other_e_embed'].unsqueeze(1).repeat(1, samle_size, 1),
                          'e_mask': history['other_e_mask'].unsqueeze(1).repeat(1, samle_size, 1)}
        post_dict = self.model.get_state_post(obs_dict=None, env_dict=None, obs_embed_dict=obs_embed_dict)
        self_h_samples = post_dict['h_dist'].sample()
        self_s_samples = belief['self_s'].repeat(samle_size, 1)

        self_h_embed = self.model.h_embed_decode_model(torch.cat([self_h_samples, self_s_samples], dim=-1))

        if action_dict['type'] == 'phy':
            if agent_name == 'robot':
                if self.env.reward_mode == 'st':
                    reward_input = torch.cat([self_h_samples, self_s_samples], dim=-1)
                else:
                    reward_input = torch.cat([self_h_samples, self_s_samples, action_dict['data'].repeat(samle_size, 1)], dim=-1)
                reward = self.model.r_reward_decode_model(reward_input).mean
            elif agent_name == 'human':
                other_zp_samples = history['other_zp'][-1].repeat(samle_size, 1)

                # TODO: Fix bug, pt_model returns other_s_embed not other_s_samples.
                # No easy way to obtain other_s_samples.
                other_s_samples = self.model.pt_model(torch.cat([self_h_embed, other_zp_samples], dim=-1)).mean
                other_s_samples = self_s_samples

                if self.env.reward_mode == 'st':
                    reward_input = torch.cat([self_h_samples, other_s_samples], dim=-1)
                else:
                    reward_input = torch.cat([self_h_samples, other_s_samples, action_dict['data'].repeat(samle_size, 1)], dim=-1)
                reward = self.model.h_reward_decode_model(reward_input).mean
            else:
                if self.env.reward_mode == 'st':
                    reward_input = torch.cat([self_h_samples, self_s_samples], dim=-1)
                else:
                    reward_input = torch.cat([self_h_samples, self_s_samples, action_dict['data'].repeat(samle_size, 1)], dim=-1)
                reward = self.model.r_reward_decode_model(reward_input).mean
            return reward.mean(dim=0).item()
        elif action_dict['type'] == 'comm':
            # communication cost
            reward = -0.5
            return reward
        else:
            # question cost
            reward = -0.5
            return reward
        # reward = -0.5
        # return reward

    def get_possible_actions(self, agent_name, type='comm', default=True):
        # agent name: who takes the action
        # type (e.g. comm, phy, query, all)
        # default: idle action, including no comm and do nothing
        if agent_name == 'robot':
            comm_actions = self.comm_generator.get_actions(agent=agent_name, device=self.model.device)
            phy_actions = self.env.get_phy_actions(agent=agent_name, device=self.model.device)
            query_actions = self.query_generator.get_actions(agent=agent_name, device=self.model.device)
            # if type == 'all':
            #     return comm_actions + phy_actions + query_actions
            # elif type == 'comm':
            #     return comm_actions
            # elif type == 'query':
            #     return query_actions
            # else:
            #     return comm_actions + phy_actions + query_actions
            return query_actions + phy_actions
        else:
            # assume human only takes phy actions
            phy_actions = self.env.get_phy_actions(agent=agent_name, device=self.model.device)
            if type == 'all':
                return phy_actions
            elif type == 'comm':
                return []
            elif type == 'query':
                return []
            else:
                return phy_actions

    def get_belief_key(self, belief):
        # belief: {'self_h_dist': distribution of h_t, 'self_s', 'self_zp', 'other_zp'}
        # belief_key = (torch.round(belief['self_h_dist'].mean * 20) / 20.0).cpu().detach().numpy()
        # belief_key = f'{belief_key}'
        # return belief_key
        belief_key = (torch.round(belief['self_s'] * 20) / 20.0).cpu().detach().numpy()
        belief_key = f'{belief_key}'
        return belief_key

    def get_action_key(self, action_dict):
        # action_dict = {'type', 'agent', 'data'}
        type_name = action_dict['type']
        agent_name = action_dict['agent']
        data_name = action_dict['data'].detach().cpu().numpy()

        return f'{type_name}_{agent_name}_{data_name}'

    def generate_comm_text(self, comm_action):
        e_embed = self.comm_generator.decoder(torch.cat([self.self_s[-1, 0], torch.tensor(comm_action).float().to(self.device)], dim=-1)).mean
        comm_text = self.hvae.comm_decode_model(e_embed)['text']
        return comm_text.cpu().detach().numpy()

    def generate_query_text(self, query_action):
        query_text = self.query_generator.decoder(torch.cat([torch.tensor(query_action).float().to(self.device), self.other_zp[-1, 0]], dim=-1)).mean
        return query_text.cpu().detach().numpy()

    def visualize_tree_node(self, node):
        num_samples = 5
        obs_embed_dict = {'s_embed': node.history['self_s_embed'].unsqueeze(1).repeat(1, num_samples, 1),
                          'e_embed': node.history['other_e_embed'].unsqueeze(1).repeat(1, num_samples, 1),
                          'e_mask': node.history['other_e_mask'].unsqueeze(1).repeat(1, num_samples, 1)}
        post_dict = self.model.get_state_post(obs_dict=None, env_dict=None, obs_embed_dict=obs_embed_dict)

        pi_obs = self.model.xh_decode_model(torch.cat([post_dict['h_dist'].sample(), node.belief['self_s'].repeat(num_samples, 1)], dim=-1))
        if 'img' in pi_obs:
            pi_obs['img'] = torch.permute(pi_obs['img'], (2, 0, 3, 1))
            pi_obs['img'] = torch.flatten(pi_obs['img'], 1, 2)
            pi_obs['img'] = torch.flip(pi_obs['img'], (-1,)) * 255
            pi_obs['img'] = pi_obs['img'].detach().cpu().numpy()

        robot_obs = self.model.xs_decode_model(node.belief['self_s'].repeat(num_samples, 1))
        if 'img' in robot_obs:
            robot_obs['img'] = torch.permute(robot_obs['img'], (2, 0, 3, 1))
            robot_obs['img'] = torch.flatten(robot_obs['img'], 1, 2)
            robot_obs['img'] = torch.flip(robot_obs['img'], (-1,)) * 255
            robot_obs['img'] = robot_obs['img'].detach().cpu().numpy()

        return pi_obs, robot_obs

    def generate_exp_data(self, timestep, env, env_config, method_config):
        # generate robot obs
        inference_net_name = 'transformer' if method_config['use_transformer'] else 'gru'
        config_id = env_config['config_id']
        config_num = env_config['config_num']

        model_id = method_config['model_id']
        method = method_config['method']
        env_name = method_config['env_name']

        folder_name = f'./testing_info/{env_name}_{model_id}_{method}_{inference_net_name}_envconfig_{config_id}_{config_num}/model_data/{timestep}'
        isExist = os.path.exists(folder_name)
        if not isExist:
            os.makedirs(folder_name)
        # ---SAVE MODEL DATA---
        robot_obs_dict = self.model.xs_decode_model(self.self_s)
        info_name = '/robot_rec_obs'
        np.savez(file=folder_name + info_name, img=torch2np(robot_obs_dict['img'], 'image'),
                 pos=torch2np(robot_obs_dict['pos'], 'state'))

        # generate robot pi obs
        robot_pi_dict = self.model.xh_decode_model(torch.cat([self.self_h, self.self_s], dim=-1))
        info_name = '/robot_rec_pi'
        np.savez(file=folder_name + info_name, img=torch2np(robot_pi_dict['img'], 'image'))

        # generate human pt obs
        self_post_dict = {'s_sample': self.self_s, 'h_sample': self.self_h}
        # human_pt_obs_dict = self.forward_other_state_post(self_post_dict)
        human_pt_obs_dict_xs = self.model.xs_decode_model(self.other_s)
        info_name = '/human_pt_obs'
        np.savez(file=folder_name + info_name, img=torch2np(human_pt_obs_dict_xs['img'], 'image'),
                 pos=torch2np(human_pt_obs_dict_xs['pos'], 'state'))
        # cv2.imshow('gen pt obs test', np.hstack(torch2np(human_pt_obs_dict_xs['img'],'image')[0]))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # generate human pi obs
        human_pt_pi_dict = self.model.xh_decode_model(torch.cat([self.other_h, self.other_s], dim=-1))
        info_name = '/human_pt_pi'
        np.savez(file=folder_name + info_name, img=torch2np(human_pt_pi_dict['img'], 'image'),
                 pos=torch2np(human_pt_obs_dict_xs['pos'], 'state'))

        # NOTE: BL of states is limited by config.batch_length
        info_name = '/robot_states'
        np.savez(file=folder_name + info_name, s=torch2np(self.self_s, 'state'), h=torch2np(self.self_h, 'state'))

        info_name = '/human_states'
        np.savez(file=folder_name + info_name, s=torch2np(self.other_s, 'state'), h=torch2np(self.other_h, 'state'))

        # ---SAVE ENV DATA---
        info_name = '/robot_info'
        num_samples = robot_obs_dict['img'].shape[1]
        np.savez(file=folder_name + info_name, img=np.repeat(env.robot_img_buffer[:, np.newaxis, ...], num_samples, axis=1),
                 pos=np.repeat(env.robot_pos_buffer[:, np.newaxis, ...], num_samples, axis=1),
                 other_comm_data=np.repeat(env.robot_other_comm_data_buffer[:, np.newaxis, ...], num_samples, axis=1),
                 other_comm_mask=np.repeat(env.robot_other_comm_mask_buffer[:, np.newaxis, ...], num_samples, axis=1),
                 action=np.repeat(env.robot_action_buffer[:, np.newaxis, ...], num_samples, axis=1))
        info_name = '/human_info'
        np.savez(file=folder_name + info_name, img=np.repeat(env.human_img_buffer[:, np.newaxis, ...], num_samples, axis=1),
                 pos=np.repeat(env.human_pos_buffer[:, np.newaxis, ...], num_samples, axis=1),
                 other_comm_data=np.repeat(env.human_other_comm_data_buffer[:, np.newaxis, ...], num_samples, axis=1),
                 other_comm_mask=np.repeat(env.human_other_comm_mask_buffer[:, np.newaxis, ...], num_samples, axis=1),
                 action=np.repeat(env.human_action_buffer[:, np.newaxis, ...], num_samples, axis=1))
        info_name = '/env_pi'
        np.savez(file=folder_name + info_name, img=np.repeat(env.pi_img_buffer[:, np.newaxis, ...], num_samples, axis=1),
                 robot_gt_belief_proba=np.repeat(env.robot_gt_belief_proba[np.newaxis, np.newaxis, ...], num_samples, axis=1),
                 human_gt_belief_proba=np.repeat(env.human_gt_belief_proba[np.newaxis, np.newaxis, ...], num_samples, axis=1))

        return None
    # def visualize(self, obs_gt, action, pi_gt, comm_gt, test=False, env_name=None):
    #     # obs_gt = {'robot_obs' robot image observation, 'human_obs'} (num_samples, feature)
    #     # action = {'robot', 'human'}
    #     # comm_gt = {'self_comm', 'other_comm'}
    #     self_s = self.self_s[-1]
    #     # self_h = self.model.h_decode_init_model_s(self_s).sample()
    #     self_h = self.self_h_dist.sample()
    #     self_zp = self.self_zp[-1]
    #
    #     other_s = self.other_s[-1]
    #     other_h = self.other_h[-1]
    #     other_zp = self.other_zp[-1]
    #
    #     # self_s_embed = self.model.s_embed_decode_model(self_s)
    #     self_s_embed = self.self_s_embed[-1]
    #     # self_h_embed = self.model.h_embed_decode_model(self_h)
    #     # robot_obs = self.model.xs_decode_model(self_s)
    #
    #     robot_obs = {}
    #     for obs_name, obs_data in obs_gt['robot_obs'].items():
    #         robot_obs[obs_name] = self.model.xs_decode_model.modality_modules[obs_name](self_s)
    #
    #     # z_comm = self.comm_generator.z_prior_dist.sample(self_s_embed.shape[:-1])
    #     # robot_comm = self.comm_generator.generate_text(self_s_embed, z_comm)
    #     #
    #     # z_query = self.query_generator.z_prior_dist.sample(self_s_embed.shape[:-1])
    #     # robot_query = self.query_generator.generate_text(other_zp, z_comm)
    #     #
    #     # other_s_embed = self.other_s_embed[-1]
    #     # other_h_embed = self.model.h_embed_decode_model(other_h)
    #
    #     # z_feedback = self.comm_generator.z_prior_dist.sample(other_s_embed.shape[:-1])
    #     # human_feedback = self.feedback_generator.generate_text(other_s_embed, robot_query)
    #
    #     human_obs = self.model.xs_decode_model(other_s)
    #     # human_obs = self.model.xs_embed_decode_model(other_s_embed)['img']
    #
    #     # chagne images to np
    #     # robot_obs = image_torch2np(robot_obs, size=(256, 256))
    #     # human_obs = image_torch2np(human_obs, size=(256, 256))
    #
    #     # print(robot_pi)
    #     # print(self.other_e_mask[:, 0])
    #
    #     # robot_obs_gt = image_torch2np(obs_gt['robot_obs'], size=(256, 256))
    #     # human_obs_gt = image_torch2np(obs_gt['human_obs'], size=(256, 256))
    #
    #     if self.reward_mode == 'st_at':
    #         reward_model_input = torch.cat(
    #             [self_h, self_s, action['robot']],
    #             dim=-1).detach()
    #     else:
    #         reward_model_input = torch.cat([self_h, self_s], dim=-1).detach()
    #     robot_reward = self.model.r_reward_decode_model(reward_model_input).mean
    #
    #     print("--------")
    #     print(f"Robot Position: {obs_gt['robot_pos']}")
    #     print(f"Human Position: {obs_gt['human_pos']}")
    #     #
    #     print(f"Robot Action: {action['robot'][0]}") #10 repeated actions, just need one
    #     print(f"Human Action: {action['human'][0]}") #10 repeated actions, just need one
    #
    #     #
    #     # print(f"Ground Truth Reward: {obs_gt['robot_reward']['data'][0]}") #10 REPEATED SAMPLES, JUST NEED ONE
    #     # print(f"Robot Reward: {robot_reward.mean()}") #take mean of the samples to declutter
    #     #
    #     reward_abs_diff = torch.nn.functional.l1_loss(robot_reward, obs_gt['robot_reward']['data'])
    #     # print(f"Reward Absolute Diff: {reward_abs_diff}")
    #
    #     print(f"Other Comm DATA: {comm_gt['other_comm']['text']['data']}")
    #     print(f"Other Comm Mask: {comm_gt['other_comm']['text']['mask']}")
    #
    #     if env_name == 'fetch_tool':
    #         robot_pi = self.model.xh_decode_model(torch.cat([self.self_h, self.self_s], dim=-1))['img']
    #         pi = torch.from_numpy(pi_gt['img']['data'])
    #         pi_abs_diff = torch.mean(torch.clamp(torch.abs(robot_pi.detach().cpu() - pi.reshape(3,64,64)), 0, 1))
    #         robot_obs_mse = torch.nn.functional.mse_loss(robot_obs['img'], obs_gt['robot_obs']['img']['data'])
    #         human_obs_mse = torch.nn.functional.mse_loss(human_obs['img'], obs_gt['human_obs']['img']['data'])
    #         robot_obs_mse += torch.nn.functional.mse_loss(robot_obs['img2'], obs_gt['robot_obs']['img2']['data'])
    #         human_obs_mse += torch.nn.functional.mse_loss(human_obs['img2'], obs_gt['human_obs']['img2']['data'])
    #         if test:
    #             robot_rec_pi_stacked = np.hstack(np.hstack(robot_pi.detach().cpu().numpy().transpose(0,1,3,4,2)))
    #             robot_rec_obs1_stacked = np.hstack(image_torch2np(robot_obs['img'],size=(64,64)))
    #             robot_rec_obs2_stacked = np.hstack(image_torch2np(robot_obs['img2'],size=(64,64)))
    #             human_rec_obs1_stacked = np.hstack(image_torch2np(human_obs['img'],size=(64,64)))
    #             human_rec_obs2_stacked = np.hstack(image_torch2np(human_obs['img2'],size=(64,64)))
    #             actual_pi = pi_gt['img']['data']
    #             actual_robot_obs1 = image_torch2np(obs_gt['robot_obs']['img']['data'], size=(64,64))[0]
    #             actual_robot_obs2 = image_torch2np(obs_gt['robot_obs']['img2']['data'], size=(64,64))[0]
    #             actual_human_obs1 = image_torch2np(obs_gt['human_obs']['img']['data'], size=(64,64))[0]
    #             actual_human_obs2 = image_torch2np(obs_gt['human_obs']['img2']['data'], size=(64,64))[0]
    #             blank_im = self.env.blank_img_obs
    #             actual_pi_stacked = actual_pi
    #             blank_im_stacked = blank_im
    #             for i in range(self.self_s.shape[0]-1):
    #                 actual_pi_stacked = np.append(actual_pi_stacked, actual_pi, axis=0)
    #                 blank_im_stacked = np.append(blank_im_stacked, blank_im, axis=0)
    #
    #             cv2.imshow('robot_actual_img1+rec_img1', np.hstack((actual_robot_obs1, blank_im, robot_rec_obs1_stacked) ))
    #             cv2.imshow('Robot Actual Obs2 + REC', np.hstack((actual_robot_obs2, blank_im, robot_rec_obs2_stacked)))
    #             cv2.imshow('actual_PI+robot_PI',np.hstack((actual_pi_stacked, blank_im_stacked, robot_rec_pi_stacked)))
    #             # cv2.imshow('Human Actual Obs1 + REC', np.hstack((actual_human_obs1, blank_im, human_rec_obs1_stacked)))
    #             # cv2.imshow('Human Actual Obs2 + REC', np.hstack((actual_human_obs1, blank_im, human_rec_obs2_stacked)))
    #             #
    #
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
    #
    #     else:
    #         robot_obs_mse = torch.nn.functional.mse_loss(robot_obs['img'], obs_gt['robot_obs']['img']['data'])
    #         human_obs_mse = torch.nn.functional.mse_loss(human_obs['img'], obs_gt['human_obs']['img']['data'])
    #         print(f"Robot Obs MSE: {robot_obs_mse}")
    #         print(f"Human Obs MSE: {human_obs_mse}")
    #
    #         robot_pi = self.model.xh_decode_model(torch.cat([self_h, self_s], dim=-1))
    #         human_pi = self.model.xh_decode_model(torch.cat([other_h, other_s], dim=-1))
    #
    #         if list(self.env.pi_info_dict.keys())[0] =='img':
    #             robot_pi = robot_pi['img']
    #             human_pi = human_pi['img']
    #             pi = torch.from_numpy(pi_gt['img']['data'])
    #             pi_abs_diff = torch.mean(torch.clamp(torch.abs(robot_pi.detach().cpu() - torch.permute(pi, (2, 0, 1))), 0, 1))
    #             print(f"PI Absolute Diff: {pi_abs_diff}")
    #
    #             cv2.imshow('Actual PI', pi_gt['img']['data'])
    #             cv2.imshow('Robot PI', np.hstack(image_torch2np(robot_pi, size=(64, 64))))
    #             cv2.imshow('Human PI', np.hstack(image_torch2np(human_pi, size=(64, 64))))
    #
    #         else:
    #             robot_pi = robot_pi['text']
    #             pi = torch.from_numpy(pi_gt['text']['data'])
    #             pi_abs_diff = torch.mean(torch.clamp(torch.abs(robot_pi.detach().cpu() - pi), 0, 1))
    #
    #             print(f"Ground Truth PI: {pi_gt['text']['data']}")
    #             print(f"Robot PI: {robot_pi}")
    #             print(f"PI Absolute Diff: {pi_abs_diff}")
    #
    #         cv2.imshow('Robot Actual Obs', np.hstack(image_torch2np(obs_gt['robot_obs']['img']['data'][0].unsqueeze(0), size=(64, 64))))
    #         cv2.imshow('Robot Obs', np.hstack(image_torch2np(robot_obs['img'], size=(64, 64))))
    #
    #         cv2.imshow('Human Actual Obs', np.hstack(image_torch2np(obs_gt['human_obs']['img']['data'][0].unsqueeze(0), size=(64, 64))))
    #         cv2.imshow('Human Obs', np.hstack(image_torch2np(human_obs['img'], size=(64, 64))))
    #
    #         if test:
    #             cv2.waitKey(0)
    #         else:
    #             cv2.waitKey(1)
    #     print("--------")
    #
    #     if not test:
    #         self.writer.add_scalar('Planning/reward_abs_diff', reward_abs_diff, self.writer_step)
    #         if obs_gt['robot_reward']['data'][0, 0] > 0:
    #             self.writer.add_scalar('Planning/pos_reward_abs_diff', reward_abs_diff, self.writer_step)
    #         else:
    #             self.writer.add_scalar('Planning/neg_reward_abs_diff', reward_abs_diff, self.writer_step)
    #
    #         if comm_gt['other_comm']['text']['mask'] == 0:
    #             self.writer.add_scalar('Planning/mask_0/pi_abs_diff', pi_abs_diff, self.writer_step)
    #         else:
    #             self.writer.add_scalar('Planning/mask_1/pi_abs_diff', pi_abs_diff, self.writer_step)
    #
    #         if list(self.env.pi_info_dict.keys())[0] =='img':
    #             pi = torch.permute(pi, (2, 0, 1))
    #             pi = torch.unsqueeze(pi, 0).expand(len(robot_pi), -1, -1, -1)
    #             self.writer.add_images(
    #                 'Planning/pi_obs',
    #                 torch.cat((pi, robot_pi.detach().cpu()), dim=-2),
    #                 self.writer_step)
    #
    #         self.writer.add_images(
    #             'Planning/img_obs',
    #             torch.cat((obs_gt['robot_obs']['img']['data'], robot_obs['img'], obs_gt['human_obs']['img']['data'], human_obs['img']), dim=-2),
    #             self.writer_step)
    #
    #         self.writer_step += 1
    #
    #     # change images to np
    #     # robot_obs = image_torch2np(robot_obs, size=(256, 256))
    #     # human_obs = image_torch2np(human_obs, size=(256, 256))
    #     #
    #     # robot_obs_gt = image_torch2np(obs_gt['robot_obs'], size=(256, 256))
    #     # human_obs_gt = image_torch2np(obs_gt['human_obs'], size=(256, 256))
    #
    #     # draw figures
    #     # visual_sample_size = 5  # human_s.shape[0]
    #
    #     # plt.close()
    #     # f, axes = plt.subplots(visual_sample_size, 8)
    #     #
    #     # for i in range(visual_sample_size):
    #     #     img1 = axes[i][0].imshow(robot_obs[i])
    #     #     axes[i][0].axis('off')
    #     #
    #     #     img2 = axes[i][1].imshow(human_obs[i])
    #     #     axes[i][1].axis('off')
    #     #
    #     #     img3 = axes[i][2].imshow(robot_obs_gt[i])
    #     #     axes[i][2].axis('off')
    #     #
    #     #     img4 = axes[i][3].imshow(human_obs_gt[i])
    #     #     axes[i][3].axis('off')
    #     #
    #     # plt.show()
    #     # img5 = axes[i][4].imshow(human_obs_est[i])
    #     # # axes[i][2].title.set_text(f'Robot Canvas Rec')
    #     # # axes[i][4].set_ylim([-1.5, 1.5])
    #     # axes[i][4].axis('off')
    #     #
    #     # img6 = axes[i][5].imshow(robot_canvas_obs_est[i])
    #     # axes[i][5].axis('off')
    #     #
    #     # # axes[i][5].title.set_text(f'Robot Canvas Rec')
    #     # # axes[i][5].set_ylim([-1.5, 1.5])
    #     #
    #     # img6 = axes[i][6].imshow(human_canvas_obs_est[i])
    #     # axes[i][6].axis('off')
    #     #
    #     # img7 = axes[i][7].imshow(human_canvas_obs_est[i])
    #     # axes[i][7].axis('off')

    def visualize(self, obs_gt, action, pi_gt, comm_gt, test=False, env_name=None):
        # obs_gt = {'robot_obs' robot image observation, 'human_obs'} (num_samples, feature)
        # action = {'robot', 'human'}
        # comm_gt = {'self_comm', 'other_comm'}
        self_s = self.self_s[-1]
        other_s = self.other_s[-1]

        self_h = self.self_h_dist.sample()
        self_zp = self.self_zp[-1]

        self_s_embed = self.self_s_embed[-1]

        print("--------")
        print(f"Robot Position: {obs_gt['robot_pos']}")
        print(f"Human Position: {obs_gt['human_pos']}")
        #
        print(f"Robot Action: {action['robot'][0]}")  # 10 repeated actions, just need one
        print(f"Human Action: {action['human'][0]}")  # 10 repeated actions, just need one

        print(f"Other Comm DATA: {comm_gt['other_comm']['text']['data']}")
        # print(f"Other Comm Mask: {self.other_e_mask.detach().cpu().numpy()[:, 0, 0]}")
        print(f"Other Comm Mask: {comm_gt['other_comm']['text']['mask']}")

        if env_name == 'fetch_tool' or env_name=='fetch_tool_unity':
            robot_obs = {}
            for obs_name, _ in obs_gt['robot_obs'].items():
                robot_obs[obs_name] = self.model.xs_decode_model.modality_modules[obs_name](self_s)

            human_obs = {}
            for obs_name, _ in obs_gt['human_obs'].items():
                human_obs[obs_name] = self.model.xs_decode_model.modality_modules[obs_name](other_s)

            robot_pi = self.model.xh_decode_model(torch.cat([self.self_h, self.self_s], dim=-1))['img']
            robot_pi2 = self.model.xh_decode_model(torch.cat([self.self_h, self.self_s], dim=-1))['img2']

            pi = torch.from_numpy(pi_gt['img']['data'])
            robot_rec_pi_stacked = np.hstack(np.hstack(robot_pi.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)))
            robot_rec_pi2_stacked = np.hstack(np.hstack(robot_pi2.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)))
            robot_rec_obs1_stacked = np.hstack(image_torch2np(robot_obs['img'], size=(64, 64)))
            robot_rec_obs2_stacked = np.hstack(image_torch2np(robot_obs['img2'], size=(64, 64)))
            human_rec_obs1_stacked = np.hstack(image_torch2np(human_obs['img'], size=(64, 64)))
            human_rec_obs2_stacked = np.hstack(image_torch2np(human_obs['img2'], size=(64, 64)))
            actual_pi = pi_gt['img']['data']
            actual_pi2 = pi_gt['img2']['data']
            actual_robot_obs1 = image_torch2np(obs_gt['robot_obs']['img']['data'], size=(64, 64))[0]
            actual_robot_obs2 = image_torch2np(obs_gt['robot_obs']['img2']['data'], size=(64, 64))[0]
            actual_human_obs1 = image_torch2np(obs_gt['human_obs']['img']['data'], size=(64, 64))[0]
            actual_human_obs2 = image_torch2np(obs_gt['human_obs']['img2']['data'], size=(64, 64))[0]
            blank_im = self.env.blank_img_obs
            actual_pi_stacked = actual_pi
            actual_pi2_stacked = actual_pi2
            blank_im_stacked = blank_im
            for i in range(self.self_s.shape[0] - 1):
                actual_pi_stacked = np.append(actual_pi_stacked, actual_pi, axis=0)
                actual_pi2_stacked = np.append(actual_pi2_stacked, actual_pi2, axis=0)
                blank_im_stacked = np.append(blank_im_stacked, blank_im, axis=0)
            cv2.imshow('Robot Actual Obs1 + REC', np.hstack((actual_robot_obs1, blank_im, robot_rec_obs1_stacked)))
            cv2.imshow('Robot Actual Obs2 + REC', np.hstack((actual_robot_obs2, blank_im, robot_rec_obs2_stacked)))
            cv2.imshow('Human Actual Obs1 + REC', np.hstack((actual_human_obs1, blank_im, human_rec_obs1_stacked)))
            cv2.imshow('Human Actual Obs2 + REC', np.hstack((actual_human_obs2, blank_im, human_rec_obs2_stacked)))
            cv2.imshow('actual_PI+robot_PI', np.hstack((actual_pi_stacked, blank_im_stacked, robot_rec_pi_stacked)))
            cv2.imshow('actual_PI2+robot_PI2', np.hstack((actual_pi2_stacked, blank_im_stacked, robot_rec_pi2_stacked)))

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif env_name == 'table_assembly':
            robot_obs = {}
            for obs_name, _ in obs_gt['robot_obs'].items():
                robot_obs[obs_name] = self.model.xs_decode_model.modality_modules[obs_name](self_s)

            human_obs = {}
            for obs_name, _ in obs_gt['human_obs'].items():
                human_obs[obs_name] = self.model.xs_decode_model.modality_modules[obs_name](other_s)

            robot_pi = self.model.xh_decode_model(torch.cat([self.self_h, self.self_s], dim=-1))['img']
            human_pi = self.model.xh_decode_model(torch.cat([self.other_h, self.other_s], dim=-1))['img']

            pi = torch.from_numpy(pi_gt['img']['data'])
            robot_rec_pi_stacked = np.hstack(np.hstack(robot_pi.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)))
            human_rec_pi_stacked = np.hstack(np.hstack(human_pi.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)))

            robot_rec_obs1_stacked = np.hstack(image_torch2np(robot_obs['img'], size=(64, 64)))
            human_rec_obs1_stacked = np.hstack(image_torch2np(human_obs['img'], size=(64, 64)))
            actual_pi = pi_gt['img']['data']
            actual_robot_obs1 = image_torch2np(obs_gt['robot_obs']['img']['data'], size=(64, 64))[0]
            actual_human_obs1 = image_torch2np(obs_gt['human_obs']['img']['data'], size=(64, 64))[0]
            blank_im = np.zeros((64,64,3))
            actual_pi_stacked = actual_pi
            blank_im_stacked = blank_im
            for i in range(self.self_s.shape[0] - 1):
                actual_pi_stacked = np.append(actual_pi_stacked, actual_pi, axis=0)
                blank_im_stacked = np.append(blank_im_stacked, blank_im, axis=0)
            cv2.imshow('Robot Actual Obs1 + REC', np.hstack((actual_robot_obs1, blank_im, robot_rec_obs1_stacked)))
            cv2.imshow('Human Actual Obs1 + REC', np.hstack((actual_human_obs1, blank_im, human_rec_obs1_stacked)))
            robot_pi_imgs = np.hstack((actual_pi_stacked, blank_im_stacked, robot_rec_pi_stacked))
            robot_pi_imgs[:,::64,:] = 0
            cv2.imshow('actual_PI+robot_PI', robot_pi_imgs)
            human_pi_imgs = np.hstack((actual_pi_stacked, blank_im_stacked, human_rec_pi_stacked))
            human_pi_imgs[:, ::64, :] = 0
            cv2.imshow('actual_PI+human_PI', human_pi_imgs)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print("--------")

    def visualize_dynamics(self):
        init_dict = self.get_init_dict()
        belief = init_dict['belief']
        history = init_dict['history']
        
        action_dicts = self.get_possible_actions(type='all', agent_name='robot')
        action_dict = action_dicts[3]

        for t in range(4):
            history, belief = self.forward_state_prior(belief, history, action_dict)

        obs_embed_dict = {
            's_embed': history['self_s_embed'].unsqueeze(1),
            'e_embed': history['other_e_embed'].unsqueeze(1),
            'e_mask': history['other_e_mask'].unsqueeze(1)}
        self_post_dict = self.model.get_state_post(obs_dict=None, env_dict=None, obs_embed_dict=obs_embed_dict)
        self_h_sample = self_post_dict['h_sample']
        self_s_sample = self_post_dict['s_sample']

        self_h_embed = self.model.h_embed_decode_model(torch.cat([self_h_sample, self_s_sample], dim=-1))
        other_s_embed = self.model.pt_model(torch.cat([self_h_embed, history['other_zp'].unsqueeze(1)], dim=-1)).mean

        obs_embed_dict = {'s_embed': other_s_embed,
                            'e_embed': history['self_e_embed'].unsqueeze(1),
                            'e_mask': history['self_e_mask'].unsqueeze(1)}
        other_post_dict = self.model.get_state_post(obs_dict=None, env_dict=None, obs_embed_dict=obs_embed_dict)
        other_s_sample = other_post_dict['s_sample']

        pi = self.model.xh_decode_model(torch.cat([self_h_sample, self_s_sample], dim=-1))
        self_obs = self.model.xs_decode_model(self_s_sample)
        other_obs = self.model.xs_decode_model(other_s_sample)

        pi['img'] = torch.squeeze(pi['img'], 1)
        cv2.imshow('Imagined PI', np.hstack(image_torch2np(pi['img'], size=(64, 64))))

        self_obs['img'] = torch.squeeze(self_obs['img'], 1)
        cv2.imshow('Imagined Self Obs', np.hstack(image_torch2np(self_obs['img'], size=(64, 64))))

        other_obs['img'] = torch.squeeze(other_obs['img'], 1)
        cv2.imshow('Imagined Other Obs', np.hstack(image_torch2np(other_obs['img'], size=(64, 64))))

        cv2.waitKey(1)

    def visualize_language(self, ):
        # obs_gt = {'robot_obs' robot image observation, 'human_obs'} (num_samples, feature)
        # action = {'robot', 'human'}
        # comm_gt = {'self_comm', 'other_comm'}
        self_s = self.self_s[-1]
        self_h = self.model.h_decode_init_model(self_s).sample()
        self_zp = self.self_zp[-1]

        other_s = self.other_s[-1]
        other_h = self.model.h_decode_init_model(other_s).sample()[-1]
        other_zp = self.other_zp[-1]

        # self_s_embed = self.model.s_embed_decode_model(self_s)
        self_s_embed = self.self_s_embed[-1]
        # self_h_embed = self.model.h_embed_decode_model(self_h)
        robot_obs = self.model.xs_decode_model(self_s_embed)['img']
        robot_pi = self.model.xh_decode_model(torch.cat([self_h, self_s], dim=-1))['text']

        z_comm = self.comm_generator.z_prior_dist.sample(self_s_embed.shape[:-1])
        robot_comm = self.comm_generator.generate_text(self_s_embed, z_comm)

        z_query = self.query_generator.z_prior_dist.sample(self_s_embed.shape[:-1])
        robot_query = self.query_generator.generate_text(other_zp, z_comm)

        other_s_embed = self.other_s_embed[-1]

        z_feedback = self.comm_generator.z_prior_dist.sample(other_s_embed.shape[:-1])
        human_feedback = self.feedback_generator.generate_text(other_s_embed, robot_query)

        human_obs = self.model.xs_decode_model(other_s_embed)['img']
        # human_pi = self.model.xh_decode_model(torch.cat([other_h, other_s], dim=-1))['text']

        # chagne images to np
        robot_obs = image_torch2np(robot_obs, size=(256, 256))
        human_obs = image_torch2np(human_obs, size=(256, 256))

        comm_rec = {}
        for comm_name, _ in self.model.comm_decode_model.modality_info_dict.items():
            comm_rec[comm_name] = self.model.comm_decode_model.modality_modules[comm_name](robot_comm)

        print(comm_rec)
