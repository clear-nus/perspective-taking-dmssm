import random
from pathlib import Path

import cv2
import numpy as np
import torch


def text_similarity(text1, text2):
    return np.square(text2 - text1).mean()


def parse_text(text, template_list):
    similarity_min = 100000.0
    opti_template = None
    for template in template_list:
        similarity = text_similarity(text, template)
        if similarity < similarity_min:
            similarity_min = similarity
            opti_template = template
    return similarity_min, opti_template


def get_nearest(a, v):
    idx = np.searchsorted(a, v, side='left')
    if idx == 0:
        return a[0]
    elif idx == len(a):
        return a[idx - 1]
    else:
        if v - a[idx - 1] < a[idx] - v:
            return a[idx - 1]
        else:
            return a[idx]


class TableAssembly:
    def __init__(self, args=None, train=False):
        # self.img_dir = 'envs/table_assembly_imgs_testing'
        self.img_dir = 'envs/table_assembly_imgs'

        # Dictionary containing images, coordinates, and hole visibility
        self.imgs = {}
        self.coords = {}
        self.is_hole_visible = {}
        for agent in ('agent', 'pi'):
            self.imgs[agent] = {}
            self.coords[agent] = {
                'agent_x': set(),
                'agent_y': set(),
                'agent_z': set(),
                'hole_x': set(),
                'hole_y': set(),
                'hole_z': set(),
                'table_x': set(),
                'table_y': set(),
            }
            self.is_hole_visible[agent] = {}

        # Load images and metadata from disk
        for path in sorted(Path(self.img_dir).glob('**/*.jpg')):
            with open(path, 'rb') as file:
                img = file.read()
                img = np.fromstring(img, np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR) / 255.0

                # Metadata is contained in file name
                # agent, agent_x, agent_y, agent_z, hole_x, hole_y, hole_z, table_x, table_y, is_hole_visible
                metadata = path.stem.split('_')
                agent = metadata[0]
                agent_x, agent_y, agent_z, hole_x, hole_y, hole_z, table_x, table_y = [
                    float(''.join([c for c in s if not c.isalpha()]))
                    for s in metadata[1:-1]]
                is_hole_visible = metadata[-1][1:] == 'True'
                key = (agent_x, agent_y, agent_z, hole_x, hole_y, hole_z, table_x, table_y)

                self.imgs[agent][key] = img

                self.coords[agent]['agent_x'].add(agent_x)
                self.coords[agent]['agent_y'].add(agent_y)
                self.coords[agent]['agent_z'].add(agent_z)
                self.coords[agent]['hole_x'].add(hole_x)
                self.coords[agent]['hole_y'].add(hole_y)
                self.coords[agent]['hole_z'].add(hole_z)
                self.coords[agent]['table_x'].add(table_x)
                self.coords[agent]['table_y'].add(table_y)

                self.is_hole_visible[agent][key] = is_hole_visible

        # Sort coordinates
        for agent in self.coords.keys():
            for key in self.coords[agent].keys():
                self.coords[agent][key] = np.asarray(sorted(self.coords[agent][key]))
        self.query_size = 2
        self.querys = {
            'none': np.array([1, 0]),
            'where': np.array([0, 1]),
        }

        self.comm_list = []
        self.comm_list.append(np.asarray((0.0, 0.0, 0.0), dtype=np.float32))
        for direction in (-1.0, 1.0):
            for magnitude in (0, 1, 2, 3):
                self.comm_list.append(np.asarray((1.0, direction, magnitude), dtype=np.float32))

        self.query_list = list(np.eye(2, dtype=int))

        # Action is vector of 4 floats: (type, x, y, z)
        # 0: idle, 1: move left/right/up/down, 2: move table left/right/down
        self.actions = []
        for action_type in (0.0, 1.0, 2.0):
            if action_type == 1.0:
                for action_x in (-0.066666666, 0.066666666):
                    action = (action_type, action_x, 0.0, 0.0)
                    self.actions.append(action)
                for action_z in (-0.066666666, 0.066666666):
                    action = (action_type, 0.0, 0.0, action_z)
                    self.actions.append(action)
            elif action_type == 2.0:
                for action_x in (-0.04, 0.04):
                    action = (action_type, action_x, 0.0, 0.0)
                    self.actions.append(action)
                for action_z in (-0.04,):
                    action = (action_type, 0.0, 0.0, action_z)
                    self.actions.append(action)
            else:
                action = (0.0, 0.0, 0.0, 0.0)
                self.actions.append(action)
        self.action_to_index = {
            action: index for index, action in enumerate(self.actions)
        }

        # Onehot Actions
        self.robot_actions = list(np.eye(len(self.actions), dtype=int))
        self.human_actions = list(np.eye(len(self.actions), dtype=int))

        # 3-dof Position, Rotation is always looking at table
        self.robot_pos = np.asarray(
            (0.0, self.coords['pi']['agent_y'][0], self.coords['pi']['agent_z'][0]))
        self.human_pos = np.asarray(
            (0.0, self.coords['pi']['agent_y'][-1], self.coords['pi']['agent_z'][0]))

        self.state = None

        self.train = train
        # {f'{modality_name}': {'modality_type', 'modality_size', 'embed_size', 'latent_size'}}
        self.agent_obs_info_dict = {'img': {'modality_type': 'image', 'modality_size': (64, 64, 3), 'embed_size': 64},
                                    'pos': {'modality_type': 'state', 'modality_size': (3,), 'embed_size': 16}}

        # self.agent_comm_info_dict = {'front_text': {'modality_type': 'state', 'modality_size': (1,), 'embed_size': 32},
        #                              'back_text': {'modality_type': 'state', 'modality_size': (1,), 'embed_size': 32}}

        self.agent_comm_info_dict = {'text': {'modality_type': 'state', 'modality_size': (3,), 'embed_size': 32}}

        self.query_info_dict = {'question': {'modality_type': 'state', 'modality_size': (2,), 'embed_size': 32}}

        self.pi_info_dict = {'img': {'modality_type': 'image', 'modality_size': (64, 64, 3), 'embed_size': 64}}
        self.pos_size = 3
        self.action_size = len(self.actions)
        self.reward_mode = 'st_at'  # either 'st_at' or 's_t'.

        if self.train:
            self.comm_rate_list = [0.5]
        else:
            self.comm_rate_list = [1.0]

        self.comm_rate = random.choice(self.comm_rate_list)

        self.robot_total_reward = 0.0
        self.human_total_reward = 0.0
        self.success = 0.0
        self.time = 0.0

        # p(x_g | x_r,1:T)
        self.num_hole_pos = len(self.coords['pi']['hole_x'])
        self.num_table_pos = len(self.coords['pi']['table_x'])
        self.num_agent_x_pos = len(self.coords['agent']['agent_x'])
        self.num_agent_y_pos = len(self.coords['agent']['agent_y'])
        self.num_agent_z_pos = len(self.coords['agent']['agent_z'])

        self.noise = 0.01

        self.robot_gt_belief_proba = np.array([(self.coords['pi']['hole_x'][0] - self.noise, self.coords['pi']['hole_x'][-1] + self.noise),
                                               (self.coords['pi']['table_x'][0] - self.noise, self.coords['pi']['table_x'][-1] + self.noise)])
        self.human_gt_belief_proba = np.array([(self.coords['pi']['hole_x'][0] - self.noise, self.coords['pi']['hole_x'][-1] + self.noise),
                                               (self.coords['pi']['table_x'][0] - self.noise, self.coords['pi']['table_x'][-1] + self.noise)])
        # p(x_h | x_g, human_pos)
        # self.robot_obs_belief_proba = np.zeros((self.num_hole_pos, self.num_table_pos,
        #                                         self.num_agent_x_pos, self.num_agent_y_pos, self.num_agent_z_pos,
        #                                         self.num_agent_x_pos, self.num_agent_y_pos, self.num_agent_z_pos,
        #                                         self.num_hole_pos, self.num_table_pos,))
        # self.human_obs_belief_proba = np.zeros((self.num_hole_pos, self.num_table_pos,
        #                                         self.num_agent_x_pos, self.num_agent_y_pos, self.num_agent_z_pos,
        #                                         self.num_agent_x_pos, self.num_agent_y_pos, self.num_agent_z_pos,
        #                                         self.num_hole_pos, self.num_table_pos,))
        # prob_val = 1 / (self.num_hole_pos * self.num_table_pos)
        # for hole in range(self.num_hole_pos):
        #     for table in range(self.num_table_pos):
        #         for x_pos in range(self.num_agent_x_pos):
        #             for y_pos in range(self.num_agent_y_pos):
        #                 for z_pos in range(self.num_agent_z_pos):
        #                     key = (self.coords['agent']['agent_x'][x_pos], self.coords['agent']['agent_y'][y_pos], self.coords['agent']['agent_z'][z_pos], self.coords['pi']['hole_x'][hole], 0.0, 0.4, self.coords['pi']['table_x'][table], 0.0)
        #                     if self.is_hole_visible['agent'][key]:
        #                         self.human_obs_belief_proba[hole][table][x_pos][y_pos][z_pos][x_pos][y_pos][z_pos][hole][table] = 1.0
        #                         self.robot_obs_belief_proba[hole][table][x_pos][y_pos][z_pos][x_pos][y_pos][z_pos][hole][table] = 1.0
        #                     else:
        #                         self.human_obs_belief_proba[hole][table][x_pos][y_pos][z_pos][x_pos][y_pos][z_pos][hole][table] = prob_val
        #                         self.robot_obs_belief_proba[hole][table][x_pos][y_pos][z_pos][x_pos][y_pos][z_pos][hole][table] = prob_val

        self.robot_img_buffer = None
        self.robot_pos_buffer = None
        self.robot_action_buffer = None
        self.robot_other_comm_data_buffer = None
        self.robot_other_comm_mask_buffer = None

        self.human_img_buffer = None
        self.human_pos_buffer = None
        self.human_action_buffer = None
        self.human_other_comm_data_buffer = None
        self.human_other_comm_mask_buffer = None

        self.pi_img_buffer = None

    def reset(self, env_config=None):

        self.robot_img_buffer = None
        self.robot_pos_buffer = None
        self.robot_action_buffer = None
        self.robot_other_comm_data_buffer = None
        self.robot_other_comm_mask_buffer = None

        self.human_img_buffer = None
        self.human_pos_buffer = None
        self.human_action_buffer = None
        self.human_other_comm_data_buffer = None
        self.human_other_comm_mask_buffer = None

        self.pi_img_buffer = None

        self.robot_gt_belief_proba = np.array([(self.coords['pi']['hole_x'][0] - self.noise, self.coords['pi']['hole_x'][-1] + self.noise),
                                               (self.coords['pi']['table_x'][0] - self.noise, self.coords['pi']['table_x'][-1] + self.noise)])
        self.human_gt_belief_proba = np.array([(self.coords['pi']['hole_x'][0] - self.noise, self.coords['pi']['hole_x'][-1] + self.noise),
                                               (self.coords['pi']['table_x'][0] - self.noise, self.coords['pi']['table_x'][-1] + self.noise)])

        self.comm_rate = random.choice(self.comm_rate_list)

        self.robot_total_reward = 0.0
        self.human_total_reward = 0.0
        self.success = 0.0
        self.time = 0.0

        # reset state
        table_pos = np.asarray((
            random.choice(self.coords['pi']['table_x']),
            random.choice(self.coords['pi']['table_y']),
            1.07))
        hole_pos = np.asarray((
            random.choice(self.coords['pi']['hole_x']),
            random.choice(self.coords['pi']['hole_y']),
            random.choice(self.coords['pi']['hole_z'])))
        self.state = {
            'table_pos': table_pos,
            'hole_pos': hole_pos,
        }
        if env_config != None:
            self.state = {
                'table_pos': np.asarray(env_config['env_state']['table_pos']),
                'hole_pos': np.asarray(env_config['env_state']['hole_pos']),
            }
        # Get robot observation
        if self.train:
            # Set higher probability of starting at a position where hole can be seen
            if random.random() < 0.5:
                robot_z = random.choice(self.coords['agent']['agent_z'][:5])
            else:
                robot_z = random.choice(self.coords['agent']['agent_z'])
            self.robot_pos = np.asarray((
                random.choice(self.coords['agent']['agent_x']),
                random.choice(self.coords['agent']['agent_y']),
                robot_z))
        else:
            self.robot_pos = np.asarray((
                random.choice(self.coords['agent']['agent_x']),
                self.coords['agent']['agent_y'][-1],
                self.coords['agent']['agent_z'][-1]))

        if env_config != None:
            self.robot_pos = np.asarray(env_config['robot_init_pos'])
        robot_img_obs = self.get_img('robot')
        robot_self_comm_text = self.get_text('robot')
        robot_query = self.get_query('robot')
        robot_reward = 0.0
        robot_done = 0.0

        # Get human observation
        if self.train:
            # Set higher probability of starting at a position where hole can be seen
            if random.random() < 0.5:
                human_z = random.choice(self.coords['agent']['agent_z'][:5])
            else:
                human_z = random.choice(self.coords['agent']['agent_z'])
            self.human_pos = np.asarray((
                random.choice(self.coords['agent']['agent_x']),
                random.choice(self.coords['agent']['agent_y']),
                human_z))
        else:
            self.human_pos = np.asarray((
                random.choice(self.coords['agent']['agent_x']),
                self.coords['agent']['agent_y'][0],
                self.coords['agent']['agent_z'][-1]))

        if env_config != None:
            self.human_pos = np.asarray(env_config['human_init_pos'])
        human_img_obs = self.get_img('human')
        human_self_comm_text = self.get_text('human')
        human_query = self.get_query('human')
        human_reward = 0.0
        human_done = 0.0

        robot_info = {'agent_obs': {'img': {'type': 'image', 'data': robot_img_obs, 'mask': 1},
                                    'pos': {'type': 'state', 'data': self.robot_pos.copy(), 'mask': 1}},
                      'other_comm': {'text': {'type': 'state', 'data': human_self_comm_text, 'mask': float(random.randrange(0, 100) < 100 * self.comm_rate)}},
                      'self_comm': {'text': {'type': 'state', 'data': robot_self_comm_text, 'mask': 1}},
                      'query': {'question': {'type': 'state', 'data': robot_query, 'mask': 1}},
                      'reward': {'type': 'state', 'data': robot_reward},
                      'done': {'type': 'state', 'data': robot_done}}

        human_info = {'agent_obs': {'img': {'type': 'image', 'data': human_img_obs, 'mask': 1},
                                    'pos': {'type': 'state', 'data': self.human_pos.copy(), 'mask': 1}},
                      'other_comm': {'text': {'type': 'state', 'data': robot_self_comm_text, 'mask': float(random.randrange(0, 100) < 100 * self.comm_rate)}},
                      'self_comm': {'text': {'type': 'state', 'data': human_self_comm_text, 'mask': 1}},
                      'query': {'question': {'type': 'state', 'data': human_query, 'mask': 1}},
                      'reward': {'type': 'state', 'data': human_reward},
                      'done': {'type': 'state', 'data': human_done}}

        if env_config != None:
            robot_info['other_comm']['text']['mask'] = 0.0
            human_info['other_comm']['text']['mask'] = 0.0

        # Get perspective invariant information
        pi = self.get_pi()
        pi_info = {'img': {'type': 'image', 'data': pi}}

        robot_action = self.sample_action(idle=True)
        human_action = self.sample_action(idle=True)
        env_info = {'robot_action': {'type': 'state', 'data': robot_action},
                    'human_action': {'type': 'state', 'data': human_action},
                    'robot_pos': {'type': 'state', 'data': self.robot_pos.copy()},
                    'human_pos': {'type': 'state', 'data': self.human_pos.copy()},
                    'robot_reward': {'type': 'state', 'data': robot_reward},
                    'human_reward': {'type': 'state', 'data': human_reward},
                    'robot_done': {'type': 'state', 'data': robot_done},
                    'human_done': {'type': 'state', 'data': human_done}}

        robot_key, human_key = self.get_nearest_key('robot'), self.get_nearest_key('human')
        robot_vis, human_vis = self.is_hole_visible['agent'][robot_key], self.is_hole_visible['agent'][human_key]
        robot_vis, human_vis = np.array([robot_vis]), np.array([human_vis])
        self.robot_img_buffer = robot_img_obs[np.newaxis, ...]
        self.robot_pos_buffer = np.concatenate([robot_key, robot_vis])[np.newaxis, ...]
        self.robot_action_buffer = robot_action[np.newaxis, ...]
        self.robot_other_comm_data_buffer = robot_info['other_comm']['text']['data'][np.newaxis, ...]
        self.robot_other_comm_mask_buffer = np.array([robot_info['other_comm']['text']['mask']])[np.newaxis, ...]

        self.human_img_buffer = human_img_obs[np.newaxis, ...]
        self.human_pos_buffer = np.concatenate([human_key, human_vis])[np.newaxis, ...]
        self.human_action_buffer = human_action[np.newaxis, ...]
        self.human_other_comm_data_buffer = human_info['other_comm']['text']['data'][np.newaxis, ...]
        self.human_other_comm_mask_buffer = np.array([human_info['other_comm']['text']['mask']])[np.newaxis, ...]

        self.pi_img_buffer = pi[np.newaxis, ...]

        self.update_robot_human_belief(robot_other_comm_text=human_self_comm_text, human_other_comm_text=robot_self_comm_text,
                                       robot_query=self.querys['none'], human_query=self.querys['none'])

        return robot_info, human_info, pi_info, env_info

    def step(self, action_dict, testing_info=None):
        robot_action = action_dict['robot']
        human_action = action_dict['human']

        self.step_action(robot_action, human_action)

        # Get robot observation
        robot_img_obs = self.get_img('robot')
        robot_self_comm_text = self.get_text('robot')
        robot_query = self.get_query('robot')
        robot_reward, robot_done = self.get_reward_done(robot_action, 'robot')

        # Get human observation
        human_img_obs = self.get_img('human')
        human_self_comm_text = self.get_text('human')
        human_query = self.get_query('human')
        human_reward, human_done = self.get_reward_done(human_action, 'human')

        robot_info = {'agent_obs': {'img': {'type': 'image', 'data': robot_img_obs, 'mask': 1},
                                    'pos': {'type': 'state', 'data': self.robot_pos.copy(), 'mask': 1}},
                      'other_comm': {'text': {'type': 'state', 'data': human_self_comm_text, 'mask': float(random.randrange(0, 100) < 100 * self.comm_rate)}},
                      'self_comm': {'text': {'type': 'state', 'data': robot_self_comm_text, 'mask': 1}},
                      'query': {'question': {'type': 'state', 'data': robot_query, 'mask': 1}},
                      'reward': {'type': 'state', 'data': robot_reward},
                      'done': {'type': 'state', 'data': robot_done}}

        human_info = {'agent_obs': {'img': {'type': 'image', 'data': human_img_obs, 'mask': 1},
                                    'pos': {'type': 'state', 'data': self.human_pos.copy(), 'mask': 1}},
                      'other_comm': {'text': {'type': 'state', 'data': robot_self_comm_text, 'mask': float(random.randrange(0, 100) < 100 * self.comm_rate)}},
                      'self_comm': {'text': {'type': 'state', 'data': human_self_comm_text, 'mask': 1}},
                      'query': {'question': {'type': 'state', 'data': human_query, 'mask': 1}},
                      'reward': {'type': 'state', 'data': human_reward},
                      'done': {'type': 'state', 'data': human_done}}

        # Get perspective invariant information
        pi = self.get_pi()
        pi_info = {'img': {'type': 'image', 'data': pi, 'mask': 1}}

        env_info = {'robot_action': {'type': 'state', 'data': robot_action},
                    'human_action': {'type': 'state', 'data': human_action},
                    'robot_pos': {'type': 'state', 'data': self.robot_pos.copy()},
                    'human_pos': {'type': 'state', 'data': self.human_pos.copy()},
                    'robot_reward': {'type': 'state', 'data': robot_reward},
                    'human_reward': {'type': 'state', 'data': human_reward},
                    'robot_done': {'type': 'state', 'data': robot_done},
                    'human_done': {'type': 'state', 'data': human_done}}

        if testing_info != None:
            robot_query, human_query = testing_info['robot_query'], testing_info['human_query']
            robot_info['query']['question']['data'] = robot_query
            human_info['query']['question']['data'] = human_query
            if robot_query[0] == 1:
                robot_info['other_comm']['text']['mask'] = 0
            elif robot_query[1] == 1:
                robot_info['other_comm']['text']['mask'] = 1
            if human_query[0] == 1:
                human_info['other_comm']['text']['mask'] = 0
            elif human_query[1] == 1:
                human_info['other_comm']['text']['mask'] = 1

        robot_key, human_key = self.get_nearest_key('robot'), self.get_nearest_key('human')
        robot_vis, human_vis = self.is_hole_visible['agent'][robot_key], self.is_hole_visible['agent'][human_key]
        robot_vis, human_vis = np.array([robot_vis]), np.array([human_vis])
        self.robot_img_buffer = np.concatenate([self.robot_img_buffer, robot_img_obs[np.newaxis, ...]], axis=0)
        self.robot_pos_buffer = np.concatenate([self.robot_pos_buffer, np.concatenate([robot_key, robot_vis])[np.newaxis, ...]], axis=0)
        self.robot_action_buffer = np.concatenate([self.robot_action_buffer, robot_action[np.newaxis, ...]], axis=0)
        self.robot_other_comm_data_buffer = np.concatenate([self.robot_other_comm_data_buffer,
                                                            robot_info['other_comm']['text']['data'][np.newaxis, ...]], axis=0)
        self.robot_other_comm_mask_buffer = np.concatenate([self.robot_other_comm_mask_buffer,
                                                            np.array([robot_info['other_comm']['text']['mask']])[np.newaxis, ...]], axis=0)

        self.human_img_buffer = np.concatenate([self.human_img_buffer, human_img_obs[np.newaxis, ...]], axis=0)
        self.human_pos_buffer = np.concatenate([self.human_pos_buffer, np.concatenate([human_key, human_vis])[np.newaxis, ...]], axis=0)
        self.human_action_buffer = np.concatenate([self.human_action_buffer, human_action[np.newaxis, ...]], axis=0)
        self.human_other_comm_data_buffer = np.concatenate([self.human_other_comm_data_buffer, human_info['other_comm']['text']['data'][np.newaxis, ...]], axis=0)
        self.human_other_comm_mask_buffer = np.concatenate([self.human_other_comm_mask_buffer, np.array([human_info['other_comm']['text']['mask']])[np.newaxis, ...]], axis=0)

        self.pi_img_buffer = np.concatenate([self.pi_img_buffer, pi[np.newaxis, ...]], axis=0)

        self.update_robot_human_belief(robot_other_comm_text=human_self_comm_text, human_other_comm_text=robot_self_comm_text,
                                       robot_query=robot_query, human_query=human_query)

        self.robot_total_reward += robot_reward
        self.human_total_reward += human_reward
        self.time += 1.0

        return robot_info, human_info, pi_info, env_info

    def reset_simulate_human(self):
        return self.reset()

    def step_simulate_human(self, action_dict):
        text_threshold = 0.2
        robot_action_type, robot_action = action_dict['robot']['type'], action_dict['robot']['data']  # {'type': phy -1 cut left, 0 do nothing, -1 cut right; comm; query, 'data'}
        human_action_type, human_action = action_dict['human']['type'], action_dict['human']['data']  # {'type': phy -1 cut left, 0 do nothing, -1 cut right; comm; query, 'data'}

        # Get robot action
        robot_self_comm_text = self.comm_list[0]
        robot_query = self.query_list[0]
        robot_phy_action = self.sample_action(idle=True)
        robot_comm_mask = 0.0
        robot_query_mask = 0.0

        if robot_action_type == 'phy':
            robot_phy_action = robot_action
        elif robot_action_type == 'comm':
            similarity_min, robot_comm = parse_text(robot_action, self.comm_list)
            if similarity_min < text_threshold:
                robot_self_comm_text = robot_comm
                robot_comm_mask = 1.0
        elif robot_action_type == 'query':
            similarity_min, robot_query = parse_text(robot_action, self.query_list)
            if similarity_min < text_threshold:
                robot_query = robot_query
                robot_query_mask = 1.0

        # Get human action
        human_self_comm_text = self.comm_list[0]
        human_query = self.query_list[0]
        human_phy_action = self.sample_action(idle=True)
        human_comm_mask = 0.0
        human_query_mask = 0.0

        if human_action_type == 'phy':
            human_phy_action = human_action
        elif human_action_type == 'comm':
            similarity_min, human_comm = parse_text(human_action, self.comm_list)
            if similarity_min < text_threshold:
                human_self_comm_text = human_comm
                human_comm_mask = 1.0
        elif human_action_type == 'query':
            similarity_min, human_query = parse_text(human_action, self.query_list)
            if similarity_min < text_threshold:
                human_query = human_query
                human_query_mask = 1.0

        self.step_action(robot_action, human_action)

        # Get robot observation
        robot_img_obs = self.get_img('robot')
        robot_reward, robot_done = self.get_reward_done(robot_phy_action, 'robot')

        # Get human observation
        human_img_obs = self.get_img('human')
        human_reward, human_done = self.get_reward_done(human_phy_action, 'human')

        robot_info = {'agent_obs': {'img': {'type': 'image', 'data': robot_img_obs, 'mask': 1},
                                    'pos': {'type': 'state', 'data': self.robot_pos.copy(), 'mask': 1}},
                      'other_comm': {'text': {'type': 'state', 'data': human_self_comm_text, 'mask': robot_query_mask}},
                      'self_comm': {'text': {'type': 'state', 'data': robot_self_comm_text, 'mask': robot_comm_mask}},
                      'query': {'question': {'type': 'state', 'data': robot_query, 'mask': robot_query_mask}},
                      'reward': {'type': 'state', 'data': robot_reward},
                      'done': {'type': 'state', 'data': robot_done}}

        human_info = {'agent_obs': {'img': {'type': 'image', 'data': human_img_obs, 'mask': 1},
                                    'pos': {'type': 'state', 'data': self.human_pos.copy(), 'mask': 1}},
                      'other_comm': {'text': {'type': 'state', 'data': robot_self_comm_text, 'mask': robot_comm_mask}},
                      'self_comm': {'text': {'type': 'state', 'data': human_self_comm_text, 'mask': human_comm_mask}},
                      'query': {'question': {'type': 'state', 'data': human_query, 'mask': human_query_mask}},
                      'reward': {'type': 'state', 'data': human_reward},
                      'done': {'type': 'state', 'data': human_done}}

        # Get perspective invariant information
        pi = self.get_pi()
        pi_info = {'img': {'type': 'image', 'data': pi, 'mask': 1}}

        env_info = {'robot_action': {'type': 'state', 'data': robot_phy_action},
                    'human_action': {'type': 'state', 'data': human_phy_action},
                    'robot_pos': {'type': 'state', 'data': self.robot_pos.copy()},
                    'human_pos': {'type': 'state', 'data': self.human_pos.copy()},
                    'robot_reward': {'type': 'state', 'data': robot_reward},
                    'human_reward': {'type': 'state', 'data': human_reward},
                    'robot_done': {'type': 'state', 'data': robot_done},
                    'human_done': {'type': 'state', 'data': human_done}}

        self.robot_total_reward += robot_reward
        self.human_total_reward += human_reward
        self.time += 1.0

        return robot_info, human_info, pi_info, env_info

    def step_action(self, robot_action, human_action):
        # Map onehot action to continuous vector
        robot_action_continuous = np.asarray(self.actions[np.argmax(robot_action)], dtype=np.float32)
        human_action_continuous = np.asarray(self.actions[np.argmax(human_action)], dtype=np.float32)

        # Robot Action
        # Action is vector of 4 floats: (type, x, y, z)
        # 0: idle, 1: move left/right/up/down, 2: move table left/right/down
        robot_action_type = np.round(robot_action_continuous[0])
        robot_action_vector = robot_action_continuous[1:]
        if robot_action_type == 1:
            robot_action_vector[1] = 0.0
            self.robot_pos += robot_action_vector
            self.robot_pos[0] = np.clip(
                self.robot_pos[0],
                self.coords['agent']['agent_x'][0],
                self.coords['agent']['agent_x'][-1])
            self.robot_pos[2] = np.clip(
                self.robot_pos[2],
                self.coords['agent']['agent_z'][0],
                self.coords['agent']['agent_z'][-1])
        elif robot_action_type == 2:
            # robot_action_vector[2] = 0.0
            self.state['table_pos'] += robot_action_vector
            self.state['table_pos'][0] = np.clip(
                self.state['table_pos'][0],
                self.coords['pi']['table_x'][0],
                self.coords['pi']['table_x'][-1])
            self.state['table_pos'][1] = np.clip(
                self.state['table_pos'][1],
                self.coords['pi']['table_y'][0],
                self.coords['pi']['table_y'][-1])

        # Human Action
        # Action is vector of 4 floats: (type, x, y, z)
        # 0: idle, 1: move left/right/up/down, 2: move table left/right/down
        human_action_type = np.round(human_action_continuous[0])
        human_action_vector = human_action_continuous[1:]
        if human_action_type == 1:
            human_action_vector[1] = 0.0
            self.human_pos += human_action_vector
            self.human_pos[0] = np.clip(
                self.human_pos[0],
                self.coords['agent']['agent_x'][0],
                self.coords['agent']['agent_x'][-1])
            self.human_pos[2] = np.clip(
                self.human_pos[2],
                self.coords['agent']['agent_z'][0],
                self.coords['agent']['agent_z'][-1])
        elif human_action_type == 2:
            # human_action_vector[2] = 0.0
            self.state['table_pos'] += human_action_vector
            self.state['table_pos'][0] = np.clip(
                self.state['table_pos'][0],
                self.coords['pi']['table_x'][0],
                self.coords['pi']['table_x'][-1])
            self.state['table_pos'][1] = np.clip(
                self.state['table_pos'][1],
                self.coords['pi']['table_y'][0],
                self.coords['pi']['table_y'][-1])

    def get_nearest_key(self, agent='robot'):
        if agent == 'robot':
            pos = self.robot_pos
            agent = 'agent'
        elif agent == 'human':
            pos = self.human_pos
            agent = 'agent'
        else:
            pos = self.robot_pos
            agent = 'pi'

        agent_x = pos[0]
        agent_y = pos[1]
        agent_z = pos[2]
        hole_x = self.state['hole_pos'][0]
        hole_y = self.state['hole_pos'][1]
        hole_z = self.state['hole_pos'][2]
        table_x = self.state['table_pos'][0]
        table_y = self.state['table_pos'][1]

        agent_x = get_nearest(self.coords[agent]['agent_x'], agent_x)
        agent_y = get_nearest(self.coords[agent]['agent_y'], agent_y)
        agent_z = get_nearest(self.coords[agent]['agent_z'], agent_z)
        hole_x = get_nearest(self.coords[agent]['hole_x'], hole_x)
        hole_y = get_nearest(self.coords[agent]['hole_y'], hole_y)
        hole_z = get_nearest(self.coords[agent]['hole_z'], hole_z)
        table_x = get_nearest(self.coords[agent]['table_x'], table_x)
        table_y = get_nearest(self.coords[agent]['table_y'], table_y)

        key = (agent_x, agent_y, agent_z, hole_x, hole_y, hole_z, table_x, table_y)
        return key

    def get_img(self, agent='robot'):
        key = self.get_nearest_key(agent)
        if agent in ('robot', 'human'):
            agent = 'agent'
        # img = cv2.imdecode(self.imgs[agent][key], cv2.IMREAD_COLOR) / 255.0
        img = self.imgs[agent][key]
        return img

    def get_text(self, agent='robot'):
        key = self.get_nearest_key(agent)
        if agent == 'robot':
            pos = self.robot_pos
            agent = 'agent'
        elif agent == 'human':
            pos = self.human_pos
            agent = 'agent'

        # Text is vector of 3 floats: (type, direction, magnitude)
        # that describes direction of hole from peg
        # 0: hole not visible, 1: hole visible
        if self.is_hole_visible[agent][key]:
            direction = self.state['hole_pos'] - self.state['table_pos']
            direction = direction[0]

            magnitude = (0.001, 0.04, 0.12, 0.2)
            magnitude_dict = {v: k for k, v in enumerate(magnitude)}
            magnitude = get_nearest(magnitude, np.abs(direction))
            magnitude = magnitude_dict[magnitude]

            text = np.asarray((1.0, np.sign(direction), magnitude))
        else:
            text = np.asarray((0.0, 0.0, 0.0))
        text = text.astype(np.float32)

        return text

    def get_query(self, agent='robot'):
        key = self.get_nearest_key(agent)
        if agent in ('robot', 'human'):
            agent = 'agent'
        if self.is_hole_visible[agent][key]:
            query = self.querys['none']
        else:
            query = self.querys['where']

        return query

    def get_pi(self):
        return self.get_img('pi')

    def get_reward_done(self, action, agent='robot'):
        idle_penalty = -0.1
        movement_penalty = -0.2
        penalty = -10.0
        bonus = 10.0

        reward = idle_penalty
        done = 0.0

        # Map onehot action to continuous vector
        action = np.asarray(self.actions[np.argmax(action)], dtype=np.float32)

        action_type = np.round(action[0])
        action_vector = action[1:]
        action_magnitude = np.linalg.norm(action_vector)

        if action_type == 1:
            if action_magnitude > 0.01:
                reward = 2 * movement_penalty
        elif action_type == 2:
            if action_magnitude > 0.01:
                reward = movement_penalty

            if action_vector[-1] < -0.01:
                distance = np.linalg.norm(self.state['hole_pos'][:2] - self.state['table_pos'][:2])
                if distance < 0.01:
                    reward = bonus
                    self.success = 1.0
                else:
                    reward = penalty
                done = 1.0

        return reward, done

    def sample_action(self, idle=False):
        # Sampled action is onehot that maps to a vector
        if idle:
            return np.asarray(self.robot_actions[0], dtype=np.float32)
        else:
            # Higher chance of sampling an optimal action
            if random.random() < 0.5:
                if random.random() < 0.75:
                    # Optimally move down
                    action = (1.0, 0.0, 0.0, -0.066666666)
                else:
                    # Optimally move table towards hole
                    correct_direction = self.state['hole_pos'][0] - self.state['table_pos'][0]
                    correct_direction = np.sign(correct_direction) * 0.04
                    if correct_direction == 0.0:
                        action = (2.0, 0.0, 0.0, -0.04)
                    else:
                        action = (2.0, correct_direction, 0.0, 0.0)
                action = self.robot_actions[self.action_to_index[action]]
                return np.asarray(action, dtype=np.float32)
            else:
                return np.asarray(random.choice(self.robot_actions), dtype=np.float32)

    def get_phy_actions(self, device, agent='robot', default=False):
        action_dicts = []
        for action_np in self.robot_actions:
            action_data = torch.as_tensor(action_np).to(device).float()
            action_dict = {'type': 'phy', 'agent': agent, 'data': action_data}
            action_dicts += [action_dict]
        return action_dicts

    def get_human_feedback(self, ):
        return None

    def get_metric(self):
        return {'robot_total_reward': self.robot_total_reward,
                'human_total_reward': self.human_total_reward,
                'success': self.success,
                'time': self.time}

    def update_robot_human_belief(self, robot_other_comm_text, human_other_comm_text, robot_query, human_query):
        magnitude = [0.001, 0.04, 0.12, 0.2]  # magnitude used in communication
        # bounds are the halved difference between the magnitudes
        bounds = [(magnitude[i + 1] - magnitude[i]) / 2 for i in range(len(magnitude) - 1)]
        mag_range = np.array([[0.0, magnitude[0] + bounds[0]],
                              [magnitude[1] - bounds[0], magnitude[1] + bounds[1]],
                              [magnitude[2] - bounds[1], magnitude[2] + bounds[2]],
                              [magnitude[3] - bounds[2], magnitude[3] + 0.05]])

        # get robot and human position
        robot_key, human_key = self.get_nearest_key('robot'), self.get_nearest_key('human')

        hole_x = self.state['hole_pos'][0]
        table_x = self.state['table_pos'][0]

        noise = self.noise

        # update robot belief states
        if self.is_hole_visible['agent'][robot_key]:
            self.robot_gt_belief_proba = np.array(([hole_x - noise, hole_x + noise], [table_x - noise, table_x + noise]))
        elif robot_other_comm_text[0] != 0 and robot_query[1] == 1:  # human sees the hole
            _, robot_direction, robot_magnitude = robot_other_comm_text.astype(np.int)
            dist = np.sort(robot_direction * mag_range[robot_magnitude])
            self.robot_gt_belief_proba = np.array(([table_x + dist[0], table_x + dist[1]], [table_x - noise, table_x + noise]))

        else:  # human doesn't see the hole
            pass

        # update human belief states
        if self.is_hole_visible['agent'][human_key]:
            self.human_gt_belief_proba = np.array(([hole_x - noise, hole_x + noise], [table_x - noise, table_x + noise]))
        elif human_other_comm_text[0] != 0 and human_query[1] == 1:  # robot sees the hole
            _, human_direction, human_magnitude = human_other_comm_text.astype(np.int)
            dist = np.sort(human_direction * mag_range[human_magnitude])
            self.human_gt_belief_proba = np.array(([table_x + dist[0], table_x + dist[1]], [table_x - noise, table_x + noise]))
        else:  # human doesn't see the hole
            pass


if __name__ == '__main__':
    env = TableAssembly(train=True)
    robot_info, human_info, pi_info, env_info = env.reset()

    print('pos', env_info['robot_pos']['data'], env_info['human_pos']['data'])

    robot_obs = robot_info['agent_obs']['img']['data']
    human_obs = human_info['agent_obs']['img']['data']
    pi = pi_info['img']['data']
    cv2.imshow('robot_obs', robot_obs)
    cv2.imshow('human_obs', human_obs)
    cv2.imshow('pi', pi)
    cv2.waitKey(50)

    robot_comm = robot_info['self_comm']['text']['data']
    human_comm = human_info['self_comm']['text']['data']
    print('robot_comm', robot_comm)
    print('human_comm', human_comm)

    while True:
        action_dict = {
            'robot': env.sample_action(),
            'human': env.sample_action(),
        }

        robot_info, human_info, pi_info, env_info = env.step(action_dict=action_dict)

        if env_info['robot_done']['data'] or env_info['human_done']['data']:
            robot_info, human_info, pi_info, env_info = env.reset()

        print('pos', env_info['robot_pos']['data'], env_info['human_pos']['data'])

        robot_obs = robot_info['agent_obs']['img']['data']
        human_obs = human_info['agent_obs']['img']['data']
        pi = pi_info['img']['data']
        cv2.imshow('robot_obs', robot_obs)
        cv2.imshow('human_obs', human_obs)
        cv2.imshow('pi', pi)

        robot_comm = robot_info['self_comm']['text']['data']
        human_comm = human_info['self_comm']['text']['data']
        print('robot_comm', robot_comm)
        print('human_comm', human_comm)

        cv2.waitKey(50)
