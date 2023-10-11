import numpy as np
import torch
import math
from envs.fetch_tool_unity import FetchToolUnity

from configs.fetch_tool_unity import *

from envs.table_assembly import TableAssembly
from configs.table_assembly import *

from agents.agent_train.hvae_ccat_train import HVAECCatAgent
from agents.agent_train.deter_rnn_train import DeterRNNAgent

from agents.agent_plan.hvae_ccat_plan import HVAECCatPlanAgent
from agents.agent_plan.deter_rnn_plan import DeterRNNPlanAgent
import argparse
from utils.tools import *
import os
import json


def model_based_train(args):
    if args.env == 'fetch_tool_unity':
        if args.train_stage == 'test':
            env = FetchToolUnity(train=False)
        else:
            if args.train_stage == 'human':
                env = FetchToolUnity(train_human=True, train=True, switch_action=False)
            else:
                env = FetchToolUnity(train_human=False, train=True, switch_action=False)
    elif args.env == 'table_assembly':
        if args.train_stage == 'test':
            env = TableAssembly(train=False)
        else:
            env = TableAssembly(train=True)
    else:
        raise NotImplementedError

    if args.method == 'hvae_ccat':
        if args.env == 'fetch_tool_unity':
            config = fetch_tool_unity_config_hvae_ccat()
        elif args.env == 'table_assembly':
            config = table_assembly_config_hvae_ccat()
        else:
            raise NotImplementedError

        agent = HVAECCatAgent(args=args, config=config, env=env, train=True)
        plan_agent = HVAECCatPlanAgent(model=agent.hvae,
                                       comm_generator=agent.comm_generator,
                                       feedback_generator=agent.feedback_generator,
                                       query_generator=agent.query_generator,
                                       env=env,
                                       inference_length=config.batch_length)
    elif args.method == 'deter_rnn':
        if args.env == 'fetch_tool_unity':
            config = fetch_tool_unity_config_deter_rnn()
        elif args.env == 'table_assembly':
            config = table_assembly_config_deter_rnn()
        else:
            raise NotImplementedError

        agent = DeterRNNAgent(args=args, config=config, env=env, train=True)
        plan_agent = DeterRNNPlanAgent(model=agent.hvae,
                                       comm_generator=agent.comm_generator,
                                       feedback_generator=agent.feedback_generator,
                                       query_generator=agent.query_generator,
                                       env=env,
                                       inference_length=config.batch_length)
    else:
        raise NotImplementedError

    inference_net_name = 'transformer' if args.use_transformer else 'gru'
    name = f'./tmp/{args.env}_{args.model_id}_{args.method}_{inference_net_name}'

    if args.train_stage == 'vae':
        config_json = json.dumps(config)
        if not os.path.exists(name):
            os.makedirs(name)
        f = open(f'{name}/config.json', 'w')
        f.write(config_json)
        f.close()
        pass
    elif args.train_stage == 'language':
        with open(f'{name}/config.json', "r") as read_file:
            config = AttrDict(json.load(read_file))
        agent.load_vae_models()
    elif args.train_stage == 'pt':
        with open(f'{name}/config.json', "r") as read_file:
            config = AttrDict(json.load(read_file))
        agent.load_vae_models()
    elif args.train_stage == 'test':
        with open(f'{name}/config.json', "r") as read_file:
            config = AttrDict(json.load(read_file))
        agent.load_vae_models()
    elif args.train_stage == 'human':
        with open(f'{name}/config.json', "r") as read_file:
            config = AttrDict(json.load(read_file))
        agent.load_vae_models()
    elif args.train_stage == 'reward':
        with open(f'{name}/config.json', "r") as read_file:
            config = AttrDict(json.load(read_file))
        agent.load_vae_models()

    agent.eval()

    episode_count = 1
    rollout_count = 1
    explore_rate = 1.01
    learning_itr_count = 0
    accumulate_robot_reward = []
    accumulate_human_reward = []

    num_samples = 10

    robot_obs_info, human_obs_info, pi_info, env_info = env.reset()

    if args.train_stage == 'human':
        role = 'human'
    else:
        role = 'robot'
    action = env.sample_action(idle=True)
    robot_action = env.sample_action(idle=True)
    human_action = env.sample_action(idle=True)

    robot_obs_info_torch = env_np2torch(robot_obs_info, num_samples=num_samples, device=agent.device)
    human_obs_info_torch = env_np2torch(human_obs_info, num_samples=num_samples, device=agent.device)
    env_info_torch = dict_np2torch(env_info, num_samples=num_samples, device=agent.device)
    action_torch = None

    while True:
        with torch.no_grad():
            if args.train_stage == 'human':
                plan_agent.forward_state_post(human_obs_info_torch, env_info_torch)
            else:
                plan_agent.forward_state_post(robot_obs_info_torch, env_info_torch)

            if role == 'robot':
                if np.random.rand(1) < 0.5:
                    action = action
                else:
                    action = env.sample_action()

                robot_action = action
                human_action = env.sample_action(idle=True)
            else:
                if args.train_stage == 'human':
                    if np.random.rand(1) < explore_rate:
                        if np.random.rand(1) < 0.5:
                            action = action
                        else:
                            action = env.sample_action()
                    else:
                        action = plan_agent.human_agent_act().detach().cpu().numpy()[0]
                else:
                    if np.random.rand(1) < 0.5:
                        action = action
                    else:
                        action = env.sample_action()

                human_action = action
                robot_action = env.sample_action(idle=True)
            action_dict = {'robot': robot_action, 'human': human_action}

            robot_obs_info, human_obs_info, pi_info, env_info = env.step(action_dict)

            robot_obs_info_torch = env_np2torch(robot_obs_info, num_samples=num_samples, device=agent.device)
            human_obs_info_torch = env_np2torch(human_obs_info, num_samples=num_samples, device=agent.device)
            env_info_torch = dict_np2torch(env_info, num_samples=num_samples, device=agent.device)
            action_torch = None

            # print(robot_action, robot_reward, done)
            accumulate_robot_reward += [robot_obs_info['reward']]
            accumulate_human_reward += [human_obs_info['reward']]

            rollout_count += 1
            if role == 'robot':
                agent.remember_robot(robot_obs_dict=robot_obs_info['agent_obs'],
                                     other_comm_dict=robot_obs_info['other_comm'],
                                     self_comm_dict=robot_obs_info['self_comm'],
                                     query_dict=robot_obs_info['query'],
                                     pi_dict=pi_info,
                                     robot_pos=env_info['robot_pos'], robot_action=env_info['robot_action'], human_action=env_info['human_action'],
                                     robot_reward=env_info['robot_reward'], robot_done=env_info['robot_done'])
            else:
                agent.remember_human(human_obs_dict=human_obs_info['agent_obs'],
                                     other_comm_dict=human_obs_info['other_comm'],
                                     self_comm_dict=human_obs_info['self_comm'],
                                     query_dict=human_obs_info['query'],
                                     pi_dict=pi_info,
                                     human_pos=env_info['human_pos'], human_action=env_info['human_action'], robot_action=env_info['robot_action'],
                                     human_reward=env_info['human_reward'], human_done=env_info['human_done'])

        if rollout_count > config.max_episode_length:
            # print(f'count: {episode_count}')
            if args.train_stage == 'human':
                role = 'human'
            else:
                if np.random.rand(1) < 0.5:
                    role = 'robot'
                else:
                    role = 'human'
            # role = 'robot'
            # print(f'role: {role} | episode: {episode_count} | robot pos: {env.robot_pos} | human pos: {env.human_pos}')
            robot_obs_info, human_obs_info, pi_info, env_info = env.reset()
            plan_agent.reset()

            robot_obs_info_torch = env_np2torch(robot_obs_info, num_samples=num_samples, device=agent.device)
            human_obs_info_torch = env_np2torch(human_obs_info, num_samples=num_samples, device=agent.device)
            env_info_torch = dict_np2torch(env_info, num_samples=num_samples, device=agent.device)
            action_torch = None

            episode_count += 1
            rollout_count = 1

        if episode_count % config.train_every == 0:
            episode_count = 1

            explore_rate *= 0.9
            if explore_rate < 0.1:
                explore_rate = 0.1

            accumulate_robot_reward.clear()
            accumulate_human_reward.clear()

            agent.train()
            if args.train_stage == 'vae':
                agent.learn(learn_itr=learning_itr_count)
            elif args.train_stage == 'pt':
                agent.learn_pt(learn_itr=learning_itr_count)

            if learning_itr_count % 50 == 0:
                if args.train_stage == 'vae':
                    agent.save_vae_models(f'{learning_itr_count}')
                elif args.train_stage == 'pt':
                    agent.save_pt_models(f'{learning_itr_count}')
                elif args.train_stage == 'reward':
                    agent.save_vae_models(f'{learning_itr_count}')
            if args.train_stage == 'vae':
                agent.save_vae_models()
            elif args.train_stage == 'pt':
                agent.save_pt_models()

            agent.eval()

            learning_itr_count += 1
            agent.to(agent.device)
            if args.train_stage == 'vae':
                n_games = config.n_games
            elif args.train_stage == 'pt':
                n_games = 100

            if learning_itr_count > n_games:
                break


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='PerspectiveTaking')
    argparser.add_argument(
        '--method',
        help='method: {hvae_ccat, deter_rnn}',
        default='hvae_transformer_cat',
        type=str)
    argparser.add_argument(
        '--env',
        help='env: {fetch_tool, table_assembly}',
        default='fetch_tool',
        type=str)
    argparser.add_argument(
        '--train_stage',
        help='train_stage: {vae, language, pt}',
        default='all',
        type=str)
    argparser.add_argument(
        "--test",
        dest='test',
        action='store_true',
        default=False)
    argparser.add_argument(
        "--use_transformer",
        help='use transformer or gru as inference net',
        default=1,
        type=int)
    argparser.add_argument(
        '--device',
        help='device',
        default=0,
        type=int)
    argparser.add_argument(
        '--model_id',
        help='model_id',
        default=10,
        type=int)
    np.set_printoptions(precision=2, suppress=True)

    args = argparser.parse_args()

    model_based_train(args)
