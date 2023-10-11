import numpy as np
import torch
import torch.nn as nn

from utils.buffer import ReplayBufferModelBased
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from agents.models.deter_rnn_model import *
from agents.models.language_generator import *
import random
from utils.tools import *


class DeterRNNAgent(nn.Module):

    def __init__(self, args=None, env=None, config=None, train=True):
        super().__init__()
        # name = f'{args.env}_{args.model_id}_{args.method}'
        inference_net_name = 'transformer' if args.use_transformer else 'gru'
        name = f'{args.env}_{args.model_id}_{args.method}_{inference_net_name}/'
        if not os.path.exists(f'./tmp/{name}'):
            os.makedirs(f'./tmp/{name}')
        device = args.device
        self.max_epoch = config.train_steps
        self.num_cat = config.num_cat
        self.robot_memory = ReplayBufferModelBased(config.max_buffer_size,
                                                   agent_obs_info_dict=env.agent_obs_info_dict,
                                                   agent_comm_info_dict=env.agent_comm_info_dict,
                                                   query_info_dict=env.query_info_dict,
                                                   pi_info_dict=env.pi_info_dict,
                                                   pos_shape=(env.pos_size,),
                                                   action_shape=(env.action_size,))
        self.human_memory = ReplayBufferModelBased(config.max_buffer_size,
                                                   agent_obs_info_dict=env.agent_obs_info_dict,
                                                   agent_comm_info_dict=env.agent_comm_info_dict,
                                                   query_info_dict=env.query_info_dict,
                                                   pi_info_dict=env.pi_info_dict,
                                                   pos_shape=(env.pos_size,),
                                                   action_shape=(env.action_size,))

        self.hvae = DeterRNN(s_size=config.s_size,
                             zp_size=config.zp_size,
                             action_size=env.action_size,
                             xp_size=env.pos_size,
                             agent_obs_info_dict=env.agent_obs_info_dict,
                             agent_comm_info_dict=env.agent_comm_info_dict,
                             pi_info_dict=env.pi_info_dict,
                             obs_fuse_embed_size=64,
                             transformer_embed_size=64,
                             name=f'{name}',
                             device=device,
                             reward_mode=env.reward_mode,
                             use_transformer=args.use_transformer)

        self.comm_generator = CommGenerator(text_size=self.hvae.e_embed_size,
                                            z_size=config.comm_action_size,
                                            s_size=self.hvae.s_embed_size,
                                            device=device)
        self.feedback_generator = FeedbackGenerator(text_size=self.hvae.e_embed_size,
                                                    z_size=config.comm_action_size,
                                                    s_size=self.hvae.s_embed_size,
                                                    query_size=env.query_info_dict['question']['modality_size'][0],
                                                    device=device)
        self.query_generator = QueryGenerator(query_size=env.query_info_dict['question']['modality_size'][0],
                                              z_size=config.query_action_size,
                                              zp_size=config.zp_size,
                                              device=device)

        self.all_models_optimizer = torch.optim.Adam(self.hvae.parameters(), lr=config.lr)
        self.pt_model_optimizer = torch.optim.Adam(self.hvae.pt_model.parameters(), lr=config.pt_lr)

        # for debug only
        self.comm_decode_optimizer = torch.optim.Adam(self.hvae.comm_decode_model.parameters(), lr=config.lr)

        # language model optimizers
        self.comm_generator_optimizer = torch.optim.Adam(self.comm_generator.parameters(), lr=config.lr)
        self.feedback_generator_optimizer = torch.optim.Adam(self.feedback_generator.parameters(), lr=config.lr)
        self.query_generator_optimizer = torch.optim.Adam(self.query_generator.parameters(), lr=config.lr)

        self.human_policy_optimizer = torch.optim.Adam(self.hvae.human_model.parameters(), lr=config.human_policy_lr)

        if not os.path.exists('runs'):
            os.makedirs('runs')

        if train:
            self.writer = SummaryWriter(log_dir=f'runs/{name}')

        self.batch_size = config.batch_size
        self.batch_length = config.batch_length

        self.device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
        self.anneal_factor = config.anneal_factor

        self.reward_mode = env.reward_mode
        self.config = config

    def remember_robot(self, robot_obs_dict, other_comm_dict, self_comm_dict, query_dict, pi_dict, robot_pos, robot_action, human_action, robot_reward,
                       robot_done):
        self.robot_memory.store_transition(robot_obs_dict, other_comm_dict, self_comm_dict, query_dict, pi_dict, robot_pos, robot_action, human_action,
                                           robot_reward, robot_done)

    def remember_human(self, human_obs_dict, other_comm_dict, self_comm_dict, query_dict, pi_dict, human_pos, human_action, robot_action, human_reward,
                       human_done):
        self.human_memory.store_transition(human_obs_dict, other_comm_dict, self_comm_dict, query_dict, pi_dict, human_pos, human_action, robot_action,
                                           human_reward, human_done)

    def save_human_policy_model(self, episode=''):
        print('.... saving human policy model ....')
        self.hvae.human_model.save_checkpoint(episode)

    def save_vae_models(self, episode=''):
        print('.... saving vae models ....')
        self.hvae.save_checkpoint(episode)

    def save_pt_models(self, episode=''):
        print('.... saving pt models ....')
        self.hvae.pt_model.save_checkpoint(episode)

    def save_language_models(self, episode=''):
        print('.... saving language models ....')
        self.comm_generator.save_checkpoint(episode)
        self.feedback_generator.save_checkpoint(episode)
        self.query_generator.save_checkpoint(episode)
        self.hvae.comm_decode_model.save_checkpoint(episode)

    def load_vae_models(self, episode=''):
        print('.... loading VAE models ....')
        self.hvae.load_checkpoint(episode)

    def load_human_policy_model(self, episode=''):
        print('.... loading human policy model ....')
        self.hvae.human_model.load_checkpoint(episode)

    def load_language_models(self, episode='', best=False):
        print('.... loading language models ....')
        self.comm_generator.load_checkpoint(episode)
        self.feedback_generator.load_checkpoint(episode)
        self.query_generator.load_checkpoint(episode)

    def load_models(self, episode='', best=False):
        print('.... loading models ....')
        self.hvae.load_checkpoint(episode)
        self.comm_generator.load_checkpoint(episode)
        self.feedback_generator.load_checkpoint(episode)
        self.query_generator.load_checkpoint(episode)

    def learn_pt(self, learn_itr):
        if self.human_memory.mem_cntr < self.batch_size * self.batch_length + 1:
            return

        if self.robot_memory.mem_cntr < self.batch_size * self.batch_length + 1:
            return
        random_batch_length = self.batch_length

        if np.random.rand(1) < 0.5:
            role = 'robot'
        else:
            role = 'human'

        if role == 'robot':
            memory = self.robot_memory
        else:
            memory = self.human_memory

        pt_losses = []
        for i in tqdm(range(self.max_epoch), desc='training'):
            agent_obs_dict, other_comm_dict, self_comm_dict, query_dict, pi_dict, pos, action, _, reward, done \
                = memory.sample_sequence_buffer(random_batch_length, self.batch_size)

            agent_obs_dict = dict_np2torch(agent_obs_dict, device=self.device)
            pi_dict = dict_np2torch(pi_dict, device=self.device)

            pos = torch.tensor(pos, dtype=torch.float).to(self.device)

            # perspective prediction
            pt_est_dict = self.hvae.perspective_predict(pi_dict, pos)
            pt_loss = 0.0
            for obs_name in agent_obs_dict:
                if obs_name != 'pos':
                    pt_loss += 100 * torch.square(pt_est_dict[obs_name] - agent_obs_dict[obs_name]['data']).mean()

            loss = pt_loss
            self.pt_model_optimizer.zero_grad()
            loss.backward()
            self.pt_model_optimizer.step()

            pt_losses += [loss.item()]
        pt_losses = torch.tensor(pt_losses).mean()
        print(f'pt_loss:{pt_losses}')
        # print(reward.detach().cpu().numpy()[:5, 0])
        # print(reward_dist.mean.detach().cpu().numpy()[:5, 0])
        # print(pi_dict['text']['data'].detach().cpu().numpy()[:5, 0])
        # print(self.hvae.xh_decode_model.modality_modules['text'](pi_embed_rec).detach().cpu().numpy()[:5, 0])
        # print('=======================================')

        pt_img = pt_est_dict['img']

        import cv2
        # visualize_images = self.hvae.xs_decode_model.modality_modules['img'](agent_embed_rec).detach().cpu().numpy().transpose((0, 1, 3, 4, 2))
        # visualize_images = agent_rec['img'].clone().detach().cpu().numpy().transpose((0, 1, 3, 4, 2))
        visualize_images2 = agent_obs_dict['img']['data'].cpu().numpy().transpose((0, 1, 3, 4, 2))
        visualize_images_pt = pt_img.detach().cpu().numpy().transpose((0, 1, 3, 4, 2))
        # visualize_images3 = canvas_obs_rec.detach().cpu().numpy().transpose((0, 1, 3, 4, 2))
        # visualize_images4 = canvas.detach().cpu().numpy().transpose((0, 1, 3, 4, 2))
        # visualize_images5 = agent_obs_rec.detach().cpu().numpy().transpose((0, 1, 3, 4, 2))
        # visualize_images6 = obs.detach().cpu().numpy().transpose((0, 1, 3, 4, 2))
        # for i in range(visualize_images_pt.shape[0]):
        #     image = np.concatenate([visualize_images2,
        #                             visualize_images_pt], axis=2)
        #     cv2.imshow(role, image[i, 0])
        #     cv2.waitKey(100)
            # print(i)

    def learn_language(self, learn_itr):
        if self.robot_memory.mem_cntr < self.batch_size * self.batch_length + 1:
            return

        memory = self.robot_memory

        for i in tqdm(range(self.max_epoch), desc='training'):
            agent_obs_dict, other_comm_dict, self_comm_dict, query_dict, pi_dict, pos, action, _, reward, done \
                = memory.sample_sequence_buffer(self.batch_length, self.batch_size)

            agent_obs_dict = dict_np2torch(agent_obs_dict, device=self.device)
            self_comm_dict = dict_np2torch(self_comm_dict, device=self.device)
            query_dict = dict_np2torch(query_dict, device=self.device)
            pos = torch.tensor(pos, dtype=torch.float).to(self.device)

            pi_dict = self.hvae.get_default_pi(pos.shape[:-1])
            agent_obs_dict.update(pi_dict)

            with torch.no_grad():
                zp_post_dist = self.hvae.zp_encode_model(pos)

                s_embed = self.hvae.s_embed_model(agent_obs_dict)
                # e_embed = self_comm_dict['text']['data']
                e_embed = self.hvae.e_embed_model(self_comm_dict)

                # todo chang to condition on s
                # transformer_embed = self.transformer(torch.cat([s_embed, e_embed * 0.0], dim=-1),
                #                                      generate_square_mask(s_embed.shape[0]).to(s_embed.device),
                #                                      src_padding_mask=None)
                #
                # s_post_dist = self.s_encode_model(transformer_embed)

            # train comm generator
            self.comm_generator_optimizer.zero_grad()
            comm_generator_loss = self.comm_generator.loss(e_state=e_embed, s_state=s_embed)
            comm_generator_loss.backward()
            self.comm_generator_optimizer.step()

            # train feedback generator
            self.feedback_generator_optimizer.zero_grad()
            feedback_generator_loss = self.feedback_generator.loss(e_state=e_embed,
                                                                   s_state=s_embed,
                                                                   query=query_dict['question']['data'])
            feedback_generator_loss.backward()
            self.feedback_generator_optimizer.step()

            self.comm_decode_optimizer.zero_grad()
            comm_decode_loss = 100 * torch.square(self.hvae.comm_decode_model(e_embed)['text'] - self_comm_dict['text']['data']).mean()
            comm_decode_loss.backward()
            self.comm_decode_optimizer.step()
            if i == self.max_epoch - 1:
                rec_e_embed = self.feedback_generator.generate_text(s_state=s_embed,
                                                                    query=query_dict['question']['data'])
                rec_comm = self.hvae.comm_decode_model(rec_e_embed)['text'].detach().cpu().numpy()
                # rec_comm = self.hvae.comm_decode_model(e_embed)['text'].detach().cpu().numpy()
                print(rec_comm)

            # train query generator
            self.query_generator_optimizer.zero_grad()
            query_generator_loss = self.query_generator.loss(query=query_dict['question']['data'], zp_state=zp_post_dist.mean)
            query_generator_loss.backward()
            self.query_generator_optimizer.step()

    def learn_human_policy(self, learn_itr):
        if self.human_memory.mem_cntr < self.batch_size * self.batch_length + 1:
            return

        random_batch_length = self.batch_length
        for i in tqdm(range(self.max_epoch), desc='training'):
            agent_obs_dict, other_comm_dict, self_comm_dict, query_dict, pi_dict, pos, action, _, reward, done \
                = self.human_memory.sample_sequence_buffer(random_batch_length, self.batch_size)

            agent_obs_dict = dict_np2torch(agent_obs_dict, device=self.device)
            other_comm_dict = dict_np2torch(other_comm_dict, device=self.device)
            self_comm_dict = dict_np2torch(self_comm_dict, device=self.device)

            pi_dict = dict_np2torch(pi_dict, device=self.device)
            # combine pi_dict and agent_obs_dict
            agent_obs_input_dict = agent_obs_dict.copy()
            agent_obs_input_dict.update(pi_dict)

            pos = torch.tensor(pos, dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.float).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            done = torch.tensor(done, dtype=torch.int).to(self.device)
            pi_dict = self.generate_masked_pi_dict(pi_dict)
            obs_dict = {'agent_obs': agent_obs_input_dict,
                        'other_comm': other_comm_dict,
                        'pi_obs': pi_dict}
            env_dict = {'pos': pos}

            post_dict = self.hvae.get_state_post(obs_dict=obs_dict, env_dict=env_dict, obs_embed_dict=None)

            # be really careful of the index, data from replay buffer (s_t, r_t, a_{t-1})
            action = action[1:]
            reward = reward[:-1]
            done = done[:-1]

            feat_sac = post_dict['s_sample'][:-1][:-1]
            action_sac = action[:-1].detach()
            reward_sac = reward[:-1].detach()
            terminal = done[:-1].type(torch.int).detach()

            # Update target networks
            update_target_networks(target_q=self.hvae.human_model.target_q_net, q=self.hvae.human_model.q_net, update_tau=0.1)
            combined_feat = torch.cat([feat_sac, action_sac], dim=-1)
            qf_pred = self.hvae.human_model.q_net(combined_feat)

            # Compute qf loss (without gradient)
            with torch.no_grad():
                target_next_feat_sac = post_dict['s_sample'][:-1][1:]
                target_value = self.hvae.human_model.compute_value(target_next_feat_sac)
                self.hvae.human_model.act(target_next_feat_sac)
                q_target = reward_sac + (1 - terminal.float()) * self.config.discount * target_value

            qf_loss = torch.nn.functional.mse_loss(qf_pred, q_target.float())

            self.human_policy_optimizer.zero_grad()
            qf_loss.backward()
            self.human_policy_optimizer.step()

    def learn(self, learn_itr):
        if self.human_memory.mem_cntr < self.batch_size * self.batch_length + 1:
            return

        if self.robot_memory.mem_cntr < self.batch_size * self.batch_length + 1:
            return

        # may change this for finetune, but now just train all the things together
        # if learn_itr <= int(1.0 * self.max_epoch):
        agent_rec_losses = []
        pi_rec_losses = []
        agent_embed_losses = []
        pi_embed_losses = []
        kl_losses = []
        reward_losses = []
        transition_losses = []

        # random_batch_length = random.randint(2, self.batch_length)
        random_batch_length = self.batch_length

        for i in tqdm(range(self.max_epoch), desc='training'):
            r_loss, r_rec_loss, r_embed_rec_loss, r_pi_loss, r_reward_rec_loss, r_transition_loss = \
                self.get_loss('robot', random_batch_length, i)
            h_loss, h_rec_loss, h_embed_rec_loss, h_pi_loss, h_reward_rec_loss, h_transition_loss = \
                self.get_loss('human', random_batch_length, i)

            self.all_models_optimizer.zero_grad()
            loss = r_loss + h_loss
            loss.backward()
            self.all_models_optimizer.step()

            agent_rec_losses += [r_rec_loss.item() + h_rec_loss.item()]
            pi_rec_losses += [r_pi_loss.item() + h_pi_loss.item()]
            agent_embed_losses += [r_embed_rec_loss.item() + h_embed_rec_loss.item()]
            reward_losses += [r_reward_rec_loss.item() + h_reward_rec_loss.item()]
            transition_losses += [r_transition_loss.item() + h_transition_loss.item()]

        self.writer.add_scalar('Training/agent_rec_loss', torch.tensor(agent_rec_losses).mean(), learn_itr)
        self.writer.add_scalar('Training/pi_rec_loss', torch.tensor(pi_rec_losses).mean(), learn_itr)
        self.writer.add_scalar('Training/agent_embed_losses', torch.tensor(agent_embed_losses).mean(), learn_itr)
        self.writer.add_scalar('Training/reward_losses', torch.tensor(reward_losses).mean(), learn_itr)
        self.writer.add_scalar('Training/transition_losses', torch.tensor(transition_losses).mean(), learn_itr)

        print(f'pi_rec_loss:{torch.tensor(pi_rec_losses).mean()}')
        print(f'agent_rec_loss:{torch.tensor(agent_rec_losses).mean()}')
        print(f'agent_embed_loss:{torch.tensor(agent_embed_losses).mean()}')
        print(f'rec_reward_loss:{torch.tensor(reward_losses).mean()}')
        print(f'transition_loss:{torch.tensor(transition_losses).mean()}')

        if self.anneal_factor > self.config.anneal_factor_end:
            self.anneal_factor *= self.config.anneal_factor_decay

    def get_loss(self, role, random_batch_length, itr_count):
        if role == 'robot':
            memory = self.robot_memory
        else:
            memory = self.human_memory

        agent_obs_dict, other_comm_dict, self_comm_dict, query_dict, pi_dict, pos, action, _, reward, done \
            = memory.sample_sequence_buffer(random_batch_length, self.batch_size)

        agent_obs_dict = dict_np2torch(agent_obs_dict, device=self.device)
        other_comm_dict = dict_np2torch(other_comm_dict, device=self.device)
        self_comm_dict = dict_np2torch(self_comm_dict, device=self.device)

        pi_dict = dict_np2torch(pi_dict, device=self.device)
        # combine pi_dict and agent_obs_dict
        agent_obs_input_dict = agent_obs_dict.copy()
        agent_obs_input_dict.update(pi_dict)

        pos = torch.tensor(pos, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.int).to(self.device)
        obs_dict = {'agent_obs': agent_obs_input_dict,
                    'other_comm': other_comm_dict,
                    'pi_obs': pi_dict}
        env_dict = {'pos': pos}

        post_dict = self.hvae.get_state_post(obs_dict=obs_dict, env_dict=env_dict, obs_embed_dict=None)

        # get the transition latent states for kl loss
        prior_dict = self.get_prior_states(random_batch_length, post_dict, action)
        transition_loss = 1.0 * torch.clip(torch.square(post_dict['s_mean'] - prior_dict['s_mean'].clone().detach()).mean(),
                                           min=0.0, max=1000000.0)
        # agent obs reconstruction
        agent_rec_loss = 0.0
        agent_rec = {}
        for obs_name, obs_data in agent_obs_dict.items():
            agent_rec[obs_name] = self.hvae.xs_decode_model.modality_modules[obs_name](post_dict['s_sample'])

            agent_rec_loss += torch.clip(self.config.agent_rec_scale[obs_name]
                                         * torch.square(agent_rec[obs_name] - obs_data['data']).mean(),
                                         min=self.config.agent_rec_lb[obs_name], max=1000000.0)

        agent_embed_rec = self.hvae.s_embed_decode_model(post_dict['s_sample'].clone().detach())
        agent_embed_rec_loss = 1.0 * torch.clip(self.config.agent_embed_rec_scale
                                                * torch.square(agent_embed_rec - post_dict['s_embed'].clone().detach()).mean(),
                                                min=0.0, max=1000000.0)

        # train for debugging only
        agent_rec_from_embed = {}
        for obs_name, obs_data in agent_obs_dict.items():
            agent_rec_from_embed[obs_name] = self.hvae.xs_embed_decode_model.modality_modules[obs_name] \
                (post_dict['s_embed'].clone().detach())

            agent_rec_loss += torch.clip(self.config.agent_rec_scale[obs_name]
                                         * torch.square(agent_rec_from_embed[obs_name] - obs_data['data']).mean(),
                                         min=self.config.agent_rec_lb[obs_name], max=1000000.0)

        # comm reconstruction (for debugging and storing history)
        comm_rec_loss = 0.0
        comm_rec = {}
        for comm_name, comm_data in other_comm_dict.items():
            comm_rec[comm_name] = self.hvae.comm_decode_model.modality_modules[comm_name](post_dict['e_embed'].clone().detach())

            comm_rec_loss += 100 * torch.square(comm_rec[comm_name] - comm_data['data']).mean().sum(-1)

        # pi obs reconstruction
        pi_rec_loss = 0.0
        pi_rec = {}
        for pi_name, pi_data in pi_dict.items():
            pi_rec[pi_name] = self.hvae.xh_decode_model.modality_modules[pi_name](post_dict['s_sample'])
            pi_rec_loss += torch.clip(self.config.pi_rec_scale[pi_name]
                                      * torch.square(pi_rec[pi_name] - pi_data['data']).mean(),
                                      min=self.config.pi_rec_lb[pi_name], max=1000000.0)

        if role == 'robot':
            reward_decode_model = self.hvae.r_reward_decode_model
            done_decode_model = self.hvae.r_done_decode_model
        else:
            reward_decode_model = self.hvae.h_reward_decode_model
            done_decode_model = self.hvae.h_done_decode_model

        if self.reward_mode == 'st_at':
            reward_model_input = torch.cat([post_dict['s_sample'][:-1], action[1:]],
                                           dim=-1).clone().detach()
            reward_dist = reward_decode_model(reward_model_input)
            reward_rec_loss = self.config.reward_rec_scale * torch.square(reward[1:] - reward_dist.mean).mean()
            done_rec_dist = done_decode_model(reward_model_input)
            done_rec_loss = - (done[1:] * (done_rec_dist.mean[..., 1:2] + 1e-5).log() +
                               (1.0 - done[1:]) * (done_rec_dist.mean[..., 0:1] + 1e-5).log()).mean()
        else:
            reward_model_input = torch.cat([post_dict['s_sample'][:-1]], dim=-1).clone().detach()
            reward_dist = reward_decode_model(reward_model_input)
            reward_rec_loss = self.config.reward_rec_scale * torch.square(reward - reward_dist.mean).mean()
            done_rec_dist = done_decode_model(reward_model_input)
            done_rec_loss = - (done * (done_rec_dist.mean[..., 1:2] + 1e-5).log() +
                               (1.0 - done) * (done_rec_dist.mean[..., 0:1] + 1e-5).log()).mean()

        # kl for transition
        # kl_trans_loss = kl_divergence_normal(post_dict['s_mean'], prior_dict['s_mean'],
        #                                      post_dict['s_std'], prior_dict['s_std'])
        # kl_trans_loss += kl_divergence_cat(post_dict['h_mean'], prior_dict['h_mean'])

        loss = agent_rec_loss + pi_rec_loss + agent_embed_rec_loss + comm_rec_loss + reward_rec_loss + done_rec_loss \
               + torch.max(transition_loss, torch.ones(1).to(transition_loss.device) * self.anneal_factor)

        # loss = kl_trans_loss
        if itr_count == self.max_epoch - 1:
            # print(reward.detach().cpu().numpy()[:5, 0])
            # print(reward_dist.mean.detach().cpu().numpy()[:5, 0])
            # print(pi_dict['text']['data'].detach().cpu().numpy()[:5, 0])
            # print(self.hvae.xh_decode_model.modality_modules['text'](pi_embed_rec).detach().cpu().numpy()[:5, 0])
            # print('=======================================')

            import cv2
            # visualize_images = self.hvae.xs_decode_model.modality_modules['img'](agent_embed_rec).detach().cpu().numpy().transpose((0, 1, 3, 4, 2))
            visualize_images = agent_rec['img'].clone().detach().cpu().numpy().transpose((0, 1, 3, 4, 2))
            visualize_images2 = agent_obs_dict['img']['data'].cpu().numpy().transpose((0, 1, 3, 4, 2))
            # visualize_images3 = canvas_obs_rec.detach().cpu().numpy().transpose((0, 1, 3, 4, 2))
            # visualize_images4 = canvas.detach().cpu().numpy().transpose((0, 1, 3, 4, 2))
            # visualize_images5 = agent_obs_rec.detach().cpu().numpy().transpose((0, 1, 3, 4, 2))
            # visualize_images6 = obs.detach().cpu().numpy().transpose((0, 1, 3, 4, 2))
            for i in range(visualize_images.shape[0]):
                image = np.concatenate([visualize_images,
                                        visualize_images2], axis=2)
                cv2.imshow('test', image[i, 0])
                cv2.waitKey(100)
                # print(i)

        return loss, agent_rec_loss, agent_embed_rec_loss, pi_rec_loss, reward_rec_loss, transition_loss

    def get_prior_states(self, random_batch_length, post_dict, action):
        s_trans_mean = [[]] * random_batch_length
        s_trans_std = [[]] * random_batch_length

        zp_trans_mean = [[]] * random_batch_length
        zp_trans_std = [[]] * random_batch_length
        for t in range(random_batch_length):
            if t == 0:
                zp_trans_mean[t] = post_dict['zp_mean'][t]

                s_trans_mean[t] = post_dict['s_mean'][t]
            else:
                _prior_dict = {'s': post_dict['s_sample'][t - 1],
                               'zp': post_dict['zp_sample'][t - 1]}
                action_dict = {'action': action[t]}
                trans_dict = self.hvae.forward_state_prior(_prior_dict, action_dict)

                zp_trans_mean[t] = trans_dict['zp_mean']

                s_trans_mean[t] = trans_dict['s_mean']

        s_trans_mean = torch.stack(s_trans_mean, dim=0)

        zp_trans_mean = torch.stack(zp_trans_mean, dim=0)

        return {'h_mean': None, 'h_sample': None,
                's_mean': s_trans_mean, 's_std': None,
                'zp_mean': zp_trans_mean, 'zp_std': None}
