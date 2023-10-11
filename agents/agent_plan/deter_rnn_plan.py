import os
import numpy as np
import torch
from POMCP.pomcp import POMCP

from utils.networks import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from agents.models.deter_rnn_model import DeterRNN
from utils.tools import *
import cv2


def get_belief_key(belief):
    # belief: {'self_z': distribution of z_t-1, 'self_zp': distribution of robot zp_t-1, 'other_zp'}
    belief_key = torch.cat([belief['self_z'], belief['self_zp'], belief['other_zp']], dim=-1).cpu().detach().numpy()
    belief_key = f'{belief_key}'
    return belief_key


class DeterRNNPlanAgent(nn.Module):
    def __init__(self,
                 model: DeterRNN,
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
        self.self_s_embed = None
        self.self_e_embed = None
        self.self_e_mask = None
        self.self_h_dist = None
        self.self_zp = None

        # estimation of human's belief of the world (time, sample_size, feature)
        self.other_s = None
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
        self.self_s_embed = None
        self.self_e_embed = None
        self.self_e_mask = None
        self.self_zp = None

        # estimation of human's belief of the world (time, sample_size, feature)
        self.other_s = None
        self.other_s_embed = None
        self.other_e_embed = None
        self.other_e_mask = None
        self.other_zp = None
        self.other_pos = None

    def forward_other_state_post(self, self_post_dict):
        # state_dict =  {'h':self_h, 'zp':other_zp}
        self_s = self_post_dict['s_sample'][-self.inference_length:]
        # h_embed = self.model.xh_embed_decode_model(self_s)
        other_pos = self.other_pos[-self.inference_length:]
        pi_dict = self.model.xh_decode_model(self_s) #is this supposed to be h_embed?

        for obs_name in self.model.pi_info_dict:
            pi_dict[obs_name] = {'modality_type': self.model.pi_info_dict[obs_name]['modality_type'], 'data': pi_dict[obs_name]}
        other_obs_dict = self.model.perspective_predict(pi_dict, other_pos)
        return other_obs_dict

    def forward_state_post(self, obs_dict, env_dict, fine_tune=False):
        # obs_dict = {'agent_obs':agent_obs_modality_dict, 'self_comm': robot comm, 'other_comm': human comm}
        # env_dict = {'robot_pos': robot pose, 'human_pos': human pose}
        # (sample_size, feature)

        # update robot states ==========================================================================================
        self_zp_post_dist = self.model.zp_encode_model(env_dict['robot_pos']['data'])
        self.self_zp = self_zp_post_dist.mean.unsqueeze(0) if self.self_zp is None else torch.cat([self.self_zp,
                                                                                                   self_zp_post_dist.mean.unsqueeze(0)], dim=0)

        # pi_dict = self.model.get_default_pi(env_dict['robot_pos']['data'].shape[:-1])
        # obs_dict['agent_obs'].update(pi_dict)
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

        self.self_s = self_post_dict['s_mean']

        self.self_s = self.self_s[-self.inference_length:]
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

            for key, val in other_obs_dict.items():
                other_obs_dict[key] = {'data': val}
            other_obs_dict['pos']['data'] = self.other_pos
            # take latest s_embed ONLY, and save to buffer!
            self.other_s_embed = self.model.s_embed_model(other_obs_dict)[-self.inference_length:]

            obs_embed_dict = {'s_embed': self.other_s_embed, 'e_embed': self.self_e_embed, 'e_mask': self.self_e_mask}
            other_post_dict = self.model.get_state_post(obs_dict=None, env_dict=None, obs_embed_dict=obs_embed_dict)

            self.other_s = other_post_dict['s_mean']

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
        # h_embed = self.model.xh_embed_decode_model(self.self_s)
        robot_pi_dict = self.model.xh_decode_model(self.self_s)
        info_name = '/robot_rec_pi'
        np.savez(file=folder_name + info_name, img=torch2np(robot_pi_dict['img'], 'image'))

        # generate human pt obs
        self_post_dict = {'s_sample': self.self_s}
        # human_pt_obs_dict = self.forward_other_state_post(self_post_dict)
        human_pt_obs_dict_xs = self.model.xs_decode_model(self.other_s)
        info_name = '/human_pt_obs'
        np.savez(file=folder_name + info_name, img=torch2np(human_pt_obs_dict_xs['img'], 'image'),
                 pos=torch2np(human_pt_obs_dict_xs['pos'], 'state'))
        # cv2.imshow('gen pt obs test', np.hstack(torch2np(human_pt_obs_dict['img'],'image')[-1]))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # generate human pi obs
        # other_h_embed = self.model.xh_embed_decode_model(self.other_s)
        human_pt_pi_dict = self.model.xh_decode_model(self.other_s)
        info_name = '/human_pt_pi'
        np.savez(file=folder_name + info_name, img=torch2np(human_pt_pi_dict['img'], 'image'),
                 pos=torch2np(human_pt_obs_dict_xs['pos'], 'state'))

        # NOTE: BL of states is limited by config.batch_length
        info_name = '/robot_states'
        np.savez(file=folder_name + info_name, s=torch2np(self.self_s, 'state'))

        info_name = '/human_states'
        np.savez(file=folder_name + info_name, s=torch2np(self.other_s, 'state'))

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

    def visualize(self, obs_gt, action, pi_gt, comm_gt, test=False, env_name=None):
        # obs_gt = {'robot_obs' robot image observation, 'human_obs'} (num_samples, feature)
        # action = {'robot', 'human'}
        # comm_gt = {'self_comm', 'other_comm'}
        self_s = self.self_s[-1]
        other_s = self.other_s[-1]

        print("--------")
        print(f"Robot Position: {obs_gt['robot_pos']}")
        print(f"Human Position: {obs_gt['human_pos']}")
        #
        print(f"Robot Action: {action['robot'][0]}")  # 10 repeated actions, just need one
        print(f"Human Action: {action['human'][0]}")  # 10 repeated actions, just need one

        print(f"Other Comm DATA: {comm_gt['other_comm']['text']['data']}")
        # print(f"Other Comm Mask: {self.other_e_mask.detach().cpu().numpy()[:, 0, 0]}")
        print(f"Other Comm Mask: {comm_gt['other_comm']['text']['mask']}")

        if env_name == 'fetch_tool' or env_name == 'fetch_tool_unity':
            robot_obs = {}
            for obs_name, _ in obs_gt['robot_obs'].items():
                robot_obs[obs_name] = self.model.xs_decode_model.modality_modules[obs_name](self_s)

            human_obs = {}
            for obs_name, _ in obs_gt['human_obs'].items():
                human_obs[obs_name] = self.model.xs_decode_model.modality_modules[obs_name](other_s)

            robot_pi = self.model.xh_decode_model(self.self_s)['img']
            robot_pi2 = self.model.xh_decode_model(self.self_s)['img2']

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
            robot_pi = self.model.xh_decode_model(self.self_s)['img']

            pi = torch.from_numpy(pi_gt['img']['data'])
            robot_rec_pi_stacked = np.hstack(np.hstack(robot_pi.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)))
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
            cv2.imshow('actual_PI+robot_PI', np.hstack((actual_pi_stacked, blank_im_stacked, robot_rec_pi_stacked)))

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print("--------")

