import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import os, cv2, glob

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


class FetchToolUnity:
    def __init__(self, train_human=False, train=False, self_input=False, switch_action=False):
        self.self_input = self_input
        self.switch_action = switch_action
        dir_path = os.path.dirname(__file__)
        instruction_imgs_path = os.path.join(dir_path, 'fetch_tool_imgs/unity/instructions/??.png')
        instruction_imgs_path_list = sorted([image for image in glob.glob(instruction_imgs_path)])
        instruction_imgs = [(file[-6:-4], cv2.resize(cv2.imread(file), (64, 64)) / 255.0) for file in instruction_imgs_path_list]
        self.instruction_imgs = dict(instruction_imgs)
        self.instruction_keys = list(self.instruction_imgs.keys())

        robot_obs_imgs_path = os.path.join(dir_path, 'fetch_tool_imgs/unity/robot_obs/??.png')
        robot_obs_imgs_path_list = sorted([image for image in glob.glob(robot_obs_imgs_path)])
        robot_obs_imgs = [(file[-6:-4], cv2.resize(cv2.imread(file), (64, 64)) / 255.0) for file in robot_obs_imgs_path_list]
        self.robot_obs_imgs = dict(robot_obs_imgs)
        self.robot_obs_keys = list(self.robot_obs_imgs.keys())

        human_obs_imgs_path = os.path.join(dir_path, 'fetch_tool_imgs/unity/human_obs/??.png')
        human_obs_imgs_path_list = sorted([image for image in glob.glob(human_obs_imgs_path)])
        human_obs_imgs = [(file[-6:-4], cv2.resize(cv2.imread(file), (64, 64)) / 255.0) for file in human_obs_imgs_path_list]
        self.human_obs_imgs = dict(human_obs_imgs)

        self.blank_img_obs = self.robot_obs_imgs['99']

        self.pi_obs2_imgs = dict()
        for key, val in self.robot_obs_imgs.items():
            self.pi_obs2_imgs[key] = cv2.rotate(val, cv2.ROTATE_90_CLOCKWISE)

        #COMMS
        #[nothing, cabinet1(00), cabinet2(01), table1(10), table2(11), machine1(22), machine2(23), cabinet, table, machine, color0, color1, color2, color3, pos]
        self.comm_size = 15
        #make one-hot vectors
        all_coms = np.arange(0, self.comm_size)
        self.comm_list = np.zeros((all_coms.size, all_coms.max() + 1), dtype=np.float32)
        self.comm_list[np.arange(all_coms.size), all_coms] = 1
        #concatenate with position
        self.comm_list = np.concatenate((self.comm_list, np.ones((15, 1))), axis=1)
        #nothing comms is only at pos 0
        self.comm_list[0, 15] = 0
        #three levels of comms
        self.cabinet_ind = 7
        self.table_ind = 8
        self.machine_ind = 9
        self.color_1_ind = 10
        self.color_2_ind = 11
        self.color_3_ind = 12
        self.color_4_ind = 13
        specific_comm_dict_values = range(1, 7)
        type_comm_dict_values = [self.cabinet_ind, self.cabinet_ind, self.table_ind, self.table_ind, self.machine_ind, self.machine_ind]
        color_comm_dict_values = [self.color_1_ind, self.color_2_ind, self.color_1_ind, self.color_2_ind, self.color_3_ind, self.color_4_ind]
        self.specific_comm_dict = dict(zip(self.instruction_keys, specific_comm_dict_values))
        self.type_comm_dict = dict(zip(self.instruction_keys, type_comm_dict_values))
        self.color_comm_dict = dict(zip(self.instruction_keys, color_comm_dict_values))

        #QUERY: [nothing, specific, type, comm]
        self.query_size = 4
        all_querys = np.arange(0, self.query_size)
        self.query_list = np.zeros((all_querys.size, all_querys.max() + 1), dtype=np.float32)
        self.query_list[np.arange(all_querys.size), all_querys] = 1
        self.human_query_list = self.query_list[1:]
        self.robot_query_list = self.query_list[0]

        self.pos_size = 1
        self.action_size = 9  # 11
        self.reward_mode = 'st_at'  # either 'st_at' or 's_t'.

        # idle, fetch tool 00, fetch tool 01, fetch tool 10, fetch tool 11, fetch tool 22, fetch tool 23, assemble, switch_pos
        all_actions = np.arange(0, self.action_size)
        self.action_list = np.zeros((all_actions.size, all_actions.max() + 1), dtype=np.float32)
        self.action_list[np.arange(all_actions.size), all_actions] = 1
        self.human_actions = self.action_list
        self.robot_actions = self.action_list

        # FRONT IS ROBOT BACK IS HUMAN
        self.num_pos = 2
        self.robot_pos = np.array([0])
        self.human_pos = np.array([1])
        # all_pos = np.arange(0, self.num_pos)
        # self.pos_list = np.zeros((all_pos.size, all_pos.max() + 1), dtype=np.float32)
        # self.pos_list[np.arange(all_pos.size), all_pos] = 1
        self.pos_list = [self.robot_pos, self.human_pos]
        self.state = None

        self.train = train
        self.train_human = train_human
        # {f'{modality_name}': {'modality_type', 'modality_size', 'embed_size', 'latent_size'}}
        self.agent_obs_info_dict = {'img': {'modality_type': 'image', 'modality_size': (64, 64, 3), 'embed_size': 32},
                                    'img2': {'modality_type': 'image', 'modality_size': (64, 64, 3), 'embed_size': 32},
                                    'pos': {'modality_type': 'state', 'modality_size': (1,), 'embed_size': 8}}

        self.agent_comm_info_dict = {'text': {'modality_type': 'state', 'modality_size': (15,), 'embed_size': 32}}

        self.query_info_dict = {'question': {'modality_type': 'state', 'modality_size': (4,), 'embed_size': 32}}

        self.pi_info_dict = {'img': {'modality_type': 'image', 'modality_size': (64, 64, 3), 'embed_size': 32},
                             'img2': {'modality_type': 'image', 'modality_size': (64, 64, 3), 'embed_size': 32}}

        if self.train:
            # self.comm_rate_list = [0.0, 0.25, 0.5]
            self.comm_rate_list = [0.5]
        else:
            self.comm_rate_list = [0.5]
        self.comm_rate = random.choice(self.comm_rate_list)
        self.robot_total_reward = 0.0
        self.human_total_reward = 0.0
        self.success = 0.0
        self.time = 0.0

        self.robot_img_buffer = None
        self.robot_img2_buffer = None
        self.robot_pos_buffer = None
        self.robot_action_buffer = None
        self.robot_other_comm_data_buffer = None
        self.robot_other_comm_mask_buffer = None

        self.human_img_buffer = None
        self.human_img2_buffer = None
        self.human_pos_buffer = None
        self.human_action_buffer = None
        self.human_other_comm_data_buffer = None
        self.human_other_comm_mask_buffer = None

        self.pi_img_buffer = None
        self.pi_img2_buffer = None

        self.robot_belief_probas = np.zeros([len(self.robot_obs_keys),len(self.instruction_keys)])
        self.robot_instruction_belief = self.instruction_keys.copy()

    def reset(self, env_config=None):
        '''
        Actions:
            [idle, fetch tool 00, fetch tool 01, fetch tool 10, fetch tool 11, fetch tool 22, fetch tool 23, assemble, switch_pos]
        Comms:
            [nothing, cabinet1(00), cabinet2(01), table1(10), table2(11), machine1(22), machine2(23), cabinet, table, machine, color0, color1, color2, color3, pos]
        Querys:
            [nothing, specific, type, comm]
        '''
        # p is the probability that the correct tool is initialized on the table
        self.robot_img_buffer =None
        self.robot_img2_buffer = None
        self.robot_pos_buffer = None
        self.robot_action_buffer = None
        self.robot_other_comm_data_buffer = None
        self.robot_other_comm_mask_buffer = None

        self.human_img_buffer =None
        self.human_img2_buffer = None
        self.human_pos_buffer = None
        self.human_action_buffer = None
        self.human_other_comm_data_buffer = None
        self.human_other_comm_mask_buffer = None

        self.pi_img_buffer = None
        self.pi_img2_buffer = None

        self.robot_instruction_belief = self.instruction_keys.copy()

        if self.train:
            r = random.uniform(0, 1)
            if r < 0.8:
                self.robot_pos = np.array([0])
                self.human_pos = np.array([1])
            else:
                self.robot_pos = np.array([1])
                self.human_pos = np.array([0])

            # r2 = random.uniform(0, 1)
            # if r2 < 0.8:
            #     self.human_pos = np.array([0])
            # else:
            #     self.human_pos = np.array([1])
        else:
            self.robot_pos = np.array([0])
            self.human_pos = np.array([1])
        self.comm_rate = random.choice(self.comm_rate_list)

        self.robot_total_reward = 0.0
        self.human_total_reward = 0.0
        self.success = 0.0
        self.time = 0.0
        # reset state
        instr = random.choice(self.instruction_keys)
        #REMEMBER TO CHANGE BACK
        rand_fetched_item = random.choice(self.instruction_keys)
        # rand_fetched_item = random.choice(self.robot_obs_keys)

        if self.self_input:
            instr = input('Enter the NEWWW Obj you want 00, 01, 10, 11, 22, 23: ')

        if self.train_human:
            p = 0.6
        else:
            p = 0.2
        if random.uniform(0, 1) < p:
            rand_fetched_item = instr

        self.state = {'instruction': instr, 'fetched_item': rand_fetched_item}

        if env_config != None:
            if env_config['robot_init_pos'] != None:
                self.robot_pos[0] = env_config['robot_init_pos']
                self.human_pos[0] = env_config['human_init_pos']
                self.state['instruction'] = env_config['env_state']['instruction']
                self.state['fetched_item'] = env_config['env_state']['fetched_item']

        if self.robot_pos == 0:  # front
            robot_img_obs = self.blank_img_obs
            robot_img_obs2 = self.robot_obs_imgs[self.state['fetched_item']]
            # robot_img_obs2 = cv2.rotate(self.instruction_imgs[self.state['fetched_item']], cv2.ROTATE_180)

            robot_text_obs, robot_query = self.create_text_query(instruction_key='99')
        else:  # back
            robot_img_obs = self.instruction_imgs[self.state['instruction']]
            robot_img_obs2 = self.human_obs_imgs[self.state['fetched_item']]
            # robot_img_obs2 = cv2.rotate(self.instruction_imgs[self.state['fetched_item']], cv2.ROTATE_180)
            robot_text_obs, robot_query = self.create_text_query(instruction_key=self.state['instruction'])
        robot_self_comm_text = np.concatenate([robot_text_obs, self.robot_pos], axis=-1)
        robot_reward = 0.0
        robot_done = 0.0

        if self.human_pos == 0:  # front
            human_img_obs = self.blank_img_obs
            human_img_obs2 = self.robot_obs_imgs[self.state['fetched_item']]
            # human_img_obs2 = self.instruction_imgs[self.state['fetched_item']]
            human_text_obs, human_query = self.create_text_query('99')
        else:
            human_img_obs = self.instruction_imgs[self.state['instruction']]
            human_img_obs2 = self.robot_obs_imgs[self.state['fetched_item']]
            # human_img_obs2 = self.instruction_imgs[self.state['fetched_item']]
            human_text_obs, human_query = self.create_text_query(self.state['instruction'])
        human_self_comm_text = np.concatenate([human_text_obs, self.human_pos], axis=-1)
        human_reward = 0.0
        human_done = 0.0

        robot_info = {'agent_obs': {'img': {'type': 'image', 'data': robot_img_obs, 'mask': 1},
                                    'img2': {'type': 'image', 'data': robot_img_obs2, 'mask': 1},
                                    'pos': {'type': 'state', 'data': self.robot_pos.copy(), 'mask': 1}},
                      'other_comm': {'text': {'type': 'state', 'data': human_self_comm_text, 'mask': 0}},
                      'self_comm': {'text': {'type': 'state', 'data': robot_self_comm_text, 'mask': 1}},
                      'query': {'question': {'type': 'state', 'data': robot_query, 'mask': 1}},
                      'reward': {'type': 'state', 'data': robot_reward},
                      'done': {'type': 'state', 'data': robot_done}}

        human_info = {'agent_obs': {'img': {'type': 'image', 'data': human_img_obs, 'mask': 1},
                                    'img2': {'type': 'image', 'data': human_img_obs2, 'mask': 1},
                                    'pos': {'type': 'state', 'data': self.human_pos.copy(), 'mask': 1}},
                      'other_comm': {'text': {'type': 'state', 'data': robot_self_comm_text, 'mask': 0}},
                      'self_comm': {'text': {'type': 'state', 'data': human_self_comm_text, 'mask': 1}},
                      'query': {'question': {'type': 'state', 'data': human_query, 'mask': 1}},
                      'reward': {'type': 'state', 'data': human_reward},
                      'done': {'type': 'state', 'data': human_done}}

        # get perspective invariant information
        pi_obs = self.instruction_imgs[self.state['instruction']]
        pi_obs2 = self.pi_obs2_imgs[self.state['fetched_item']]

        pi_info = {'img': {'type': 'image', 'data': pi_obs, 'mask': 1},
                   'img2': {'type': 'image', 'data': pi_obs2, 'mask': 1}}

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

        self.robot_img_buffer = robot_img_obs[np.newaxis,...]
        self.robot_img2_buffer = robot_img_obs2[np.newaxis,...]
        self.robot_pos_buffer = self.robot_pos.copy()[np.newaxis,...]
        self.robot_action_buffer = robot_action[np.newaxis,...]
        self.robot_other_comm_data_buffer = robot_info['other_comm']['text']['data'][np.newaxis,...]
        self.robot_other_comm_mask_buffer = np.array([robot_info['other_comm']['text']['mask']])[np.newaxis,...]

        self.human_img_buffer = human_img_obs[np.newaxis,...]
        self.human_img2_buffer = human_img_obs2[np.newaxis,...]
        self.human_pos_buffer = self.human_pos.copy()[np.newaxis,...]
        self.human_action_buffer = human_action[np.newaxis,...]
        self.human_other_comm_data_buffer = human_info['other_comm']['text']['data'][np.newaxis,...]
        self.human_other_comm_mask_buffer = np.array([human_info['other_comm']['text']['mask']])[np.newaxis,...]

        self.pi_img_buffer = pi_obs[np.newaxis,...]
        self.pi_img2_buffer = pi_obs2[np.newaxis,...]

        self.update_robot_belief(query=robot_query)

        return robot_info, human_info, pi_info, env_info

    def reset_simulate_human(self):
        # reset state
        if self.train:
            self.robot_pos = random.choice([np.array([0]), np.array([1])])
            self.human_pos = random.choice([np.array([0]), np.array([1])])
        else:
            self.robot_pos = np.array([0])
            self.human_pos = np.array([1])
        instr = random.choice(self.human_instr_ind[:-1])
        self.state = {'instruction': instr, 'fetched_item': '99'}

        # get robot observation

        if self.robot_pos == 0:  # front
            robot_img_obs = self.blank_img_obs
            robot_img_obs2 = self.human_instr_imgs[self.state['fetched_item']]
            robot_text_obs, robot_query_val = create_text_query('None', OH=self.comms_OH)
            robot_query = np.array((robot_query_val[0], 0))

        else:  # back
            robot_img_obs = self.human_instr_imgs[self.state['instruction']]
            robot_img_obs2 = self.human_instr_imgs[self.state['fetched_item']]
            robot_text_obs, robot_query_val = create_text_query(self.state['instruction'], OH=self.comms_OH)
            robot_query = np.array((0, robot_query_val[0]))

        robot_self_comm_text = np.concatenate([robot_text_obs, self.robot_pos], axis=-1)
        robot_reward = 0.0
        robot_done = 0.0

        if self.human_pos == 0:  # front
            human_img_obs = self.blank_img_obs
            human_img_obs2 = self.human_instr_imgs[self.state['fetched_item']]
            human_text_obs, human_query_val = create_text_query('None', OH=self.comms_OH)
            human_query = np.array((human_query_val[0], 0))
        else:
            human_img_obs = self.human_instr_imgs[self.state['instruction']]
            human_img_obs2 = self.human_instr_imgs[self.state['fetched_item']]
            human_text_obs, human_query_val = create_text_query(self.state['instruction'], OH=self.comms_OH)
            human_query = np.array((0, human_query_val[0]))

        human_self_comm_text = np.concatenate([human_text_obs, self.human_pos], axis=-1)
        human_reward = 0.0
        human_done = 0.0

        robot_info = {'agent_obs': {'img': {'type': 'image', 'data': robot_img_obs, 'mask': 1},
                                    'img2': {'type': 'image', 'data': robot_img_obs2, 'mask': 1},
                                    'pos': {'type': 'state', 'data': self.robot_pos.copy(), 'mask': 1}},
                      'other_comm': {'text': {'type': 'state', 'data': human_self_comm_text, 'mask': 0.0}},
                      'self_comm': {'text': {'type': 'state', 'data': robot_self_comm_text, 'mask': 0.0}},
                      'query': {'question': {'type': 'state', 'data': robot_query, 'mask': 0.0}},
                      'reward': {'type': 'state', 'data': robot_reward},
                      'done': {'type': 'state', 'data': robot_done}}

        human_info = {'agent_obs': {'img': {'type': 'image', 'data': human_img_obs, 'mask': 1},
                                    'img2': {'type': 'image', 'data': human_img_obs2, 'mask': 1},
                                    'pos': {'type': 'state', 'data': self.human_pos.copy(), 'mask': 1}},
                      'other_comm': {'text': {'type': 'state', 'data': robot_self_comm_text, 'mask': 0.0}},
                      'self_comm': {'text': {'type': 'state', 'data': human_self_comm_text, 'mask': 0.0}},
                      'query': {'question': {'type': 'state', 'data': human_query, 'mask': 0.0}},
                      'reward': {'type': 'state', 'data': human_reward},
                      'done': {'type': 'state', 'data': human_done}}

        # get perspective invariant information
        pi_obs = self.human_instr_imgs[self.state['instruction']]
        pi_obs2 = self.human_instr_imgs[self.state['fetched_item']]
        pi_info = {'img': {'type': 'image', 'data': pi_obs}, 'img2': {'type': 'image', 'data': pi_obs2} }
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

        return robot_info, human_info, pi_info, env_info

    def step_simulate_human(self, action_dict, role='robot'):
        text_threshold = 0.2
        action_type, action = action_dict['type'], action_dict['data']
        # human_action_type, human_action = action_dict['human']['type'], action_dict['human']['data']  # {'type': phy -1 cut left, 0 do nothing, -1 cut right; comm; query, 'data'}
        human_action_type, human_action = 'phy', self.sample_action(idle=True)
        # simulate human taking the optimal action first

        #DEFAULT ROBOT INFO VALUES:
        robot_text_obs, robot_query = self.create_text_query(instruction_key='99')
        robot_self_comm_text = np.concatenate([robot_text_obs, self.robot_pos], axis=-1)
        robot_query = self.query_list[0]
        robot_comm_mask = 0.0
        robot_query_mask = 0.0
        robot_phy_action = self.sample_action(idle=True)
        #DEFAULT HUMAN VALUES
        human_text_obs, human_query = self.create_text_query(instruction_key=self.state['instruction'])
        human_self_comm_text = np.concatenate([human_text_obs, self.human_pos], axis=-1)
        human_comm_mask = 0.0
        human_query_mask = 0.0
        human_phy_action = self.sample_action(idle=True)
        human_query = random.choice(self.query_list)
        if role == 'robot':
            if action_type == 'phy':
                robot_phy_action = action
                robot_action_ind = np.where(action == 1)[0][0]
                if 1 <= robot_action_ind <= 6 and self.robot_pos == 0:
                    self.state['fetched_item'] = self.instruction_keys[robot_action_ind - 1]

            elif action_type == 'comm': #remove???
                # similarity_min, robot_comm = parse_text(action, self.comm_list)
                # if similarity_min < text_threshold:
                robot_comm = action
                robot_self_comm_text = robot_comm
                robot_comm_mask = 1.0

            elif action_type == 'query':
                # QUERY: [nothing, specific, type, comm]
                # similarity_min, robot_query = parse_text(action, self.query_list)
                # if similarity_min < text_threshold:
                robot_query = action
                robot_query_mask = 1.0
                human_comm_mask = 1.0

        elif role == 'human':
            human_phy_action = action
            #NO TRANSITION
            pass

        human_reward, human_done = self.get_reward_done(human_phy_action, self.human_pos)
        robot_reward, robot_done = self.get_reward_done(robot_phy_action, self.robot_pos)

        done = human_done or robot_done
        human_done = done
        robot_done = done

        #GET HUMAN OBS
        if self.human_pos == 0:  # front
            human_img_obs = self.blank_img_obs
            human_img_obs2 = self.robot_obs_imgs[self.state['fetched_item']]
        else:
            human_img_obs = self.instruction_imgs[self.state['instruction']]
            human_img_obs2 = self.human_obs_imgs[self.state['fetched_item']]
        if robot_query_mask == 1.0: #if robot asks question, humans answer accordingly
            human_text_obs, _ = self.create_text_query(self.state['instruction'], robot_query)
            human_self_comm_text = np.concatenate([human_text_obs, self.human_pos], axis=-1)

        # get robot info
        if self.robot_pos == 0:  # front
            robot_img_obs = self.blank_img_obs
            robot_img_obs2 = self.robot_obs_imgs[self.state['fetched_item']]
        else:  # back
            robot_img_obs = self.instruction_imgs[self.state['instruction']]
            robot_img_obs2 = self.human_obs_imgs[self.state['fetched_item']]

        # if np.random.rand(1) < self.comm_rate:
        #     robot_query_mask = 1.0
        # else:
        #     robot_query_mask = 0.0

        robot_info = {'agent_obs': {'img': {'type': 'image', 'data': robot_img_obs, 'mask': 1},
                                    'img2': {'type': 'image', 'data': robot_img_obs2, 'mask': 1},
                                    'pos': {'type': 'state', 'data': self.robot_pos.copy(), 'mask': 1}},
                      'other_comm': {'text': {'type': 'state', 'data': human_self_comm_text, 'mask': robot_query_mask}},
                      'self_comm': {'text': {'type': 'state', 'data': robot_self_comm_text, 'mask': robot_comm_mask}},
                      'query': {'question': {'type': 'state', 'data': robot_query, 'mask': robot_query_mask}},
                      'reward': {'type': 'state', 'data': robot_reward},
                      'done': {'type': 'state', 'data': robot_done}}

        human_info = {'agent_obs': {'img': {'type': 'image', 'data': human_img_obs, 'mask': 1},
                                    'img2': {'type': 'image', 'data': human_img_obs2, 'mask': 1},
                                    'pos': {'type': 'state', 'data': self.human_pos.copy(), 'mask': 1}},
                      'other_comm': {'text': {'type': 'state', 'data': robot_self_comm_text, 'mask': robot_comm_mask}},
                      'self_comm': {'text': {'type': 'state', 'data': human_self_comm_text, 'mask': human_comm_mask}},
                      'query': {'question': {'type': 'state', 'data': human_query, 'mask': human_query_mask}},
                      'reward': {'type': 'state', 'data': human_reward},
                      'done': {'type': 'state', 'data': human_done}}

        # get perspective invariant information
        pi_obs = self.instruction_imgs[self.state['instruction']]
        pi_obs2 = self.pi_obs2_imgs[self.state['fetched_item']]
        pi_info = {'img': {'type': 'image', 'data': pi_obs, 'mask': 1},
                   'img2': {'type': 'image', 'data': pi_obs2, 'mask': 1} }
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
        if robot_reward > 0.0:
            self.success = 1.0
        else:
            self.success = 0.0
        self.time += 1.0

        return robot_info, human_info, pi_info, env_info

    def step(self, action_dict, testing_info=None):
        '''
        Actions:
            [idle, fetch tool 00, fetch tool 01, fetch tool 10, fetch tool 11, fetch tool 22, fetch tool 23, assemble, switch_pos]
        Comms:
            [nothing, cabinet1(00), cabinet2(01), table1(10), table2(11), machine1(22), machine2(23), cabinet, table, machine, color0, color1, color2, color3, pos]
        Querys:
            [nothing, specific, type, comm]
        '''
        robot_action = action_dict['robot']
        human_action = action_dict['human']

        robot_action_ind = np.where(robot_action == 1)[0][0]
        human_action_ind = np.where(human_action == 1)[0][0]

        # switch positions
        if self.switch_action:
            if robot_action_ind == 8:
                self.robot_pos = np.array([0]) if self.robot_pos == 1 else np.array([1])
            if human_action_ind == 8:  # 10:
                self.human_pos = np.array([0]) if self.human_pos == 1 else np.array([1])

        # if both fetches item, human will fetch item first, then robot will replace it
        if 1 <= human_action_ind <= 6 and self.human_pos == 0:
            self.state['fetched_item'] = self.instruction_keys[human_action_ind - 1]

        human_reward, human_done = self.get_reward_done(human_action, self.human_pos)

        if 1 <= robot_action_ind <= 6 and self.robot_pos == 0:
            self.state['fetched_item'] = self.instruction_keys[robot_action_ind - 1]
        robot_reward, robot_done = self.get_reward_done(robot_action, self.robot_pos)
        # game ends when anyone chooses to assemble
        done = robot_done or human_done
        robot_done = done
        human_done = done

        # get robot info
        if self.robot_pos == 0:  # front
            robot_img_obs = self.blank_img_obs
            robot_img_obs2 = self.robot_obs_imgs[self.state['fetched_item']]
            # robot_img_obs2 = cv2.rotate(self.instruction_imgs[self.state['fetched_item']], cv2.ROTATE_180)
            robot_text_obs, robot_query = self.create_text_query(instruction_key='99')
        else:  # back
            robot_img_obs = self.instruction_imgs[self.state['instruction']]
            robot_img_obs2 = self.human_obs_imgs[self.state['fetched_item']]
            # robot_img_obs2 = cv2.rotate(self.instruction_imgs[self.state['fetched_item']], cv2.ROTATE_180)
            robot_text_obs, robot_query = self.create_text_query(instruction_key=self.state['instruction'])
        robot_self_comm_text = np.concatenate([robot_text_obs, self.robot_pos], axis=-1)

        # get human info
        if self.human_pos == 0:  # front
            human_img_obs = self.blank_img_obs
            human_img_obs2 = self.robot_obs_imgs[self.state['fetched_item']]
            # human_img_obs2 = self.instruction_imgs[self.state['fetched_item']]
            human_text_obs, human_query = self.create_text_query('99')
        else:
            human_img_obs = self.instruction_imgs[self.state['instruction']]
            human_img_obs2 = self.human_obs_imgs[self.state['fetched_item']]
            # human_img_obs2 = self.instruction_imgs[self.state['fetched_item']]
            human_text_obs, human_query = self.create_text_query(self.state['instruction'])
        human_self_comm_text = np.concatenate([human_text_obs, self.human_pos], axis=-1)



        robot_info = {'agent_obs': {'img': {'type': 'image', 'data': robot_img_obs, 'mask': 1},
                                    'img2': {'type': 'image', 'data': robot_img_obs2, 'mask': 1},
                                    'pos': {'type': 'state', 'data': self.robot_pos.copy(), 'mask': 1}},
                      'other_comm': {'text': {'type': 'state', 'data': human_self_comm_text, 'mask': float(random.randrange(0, 100) < 100 * self.comm_rate)}},
                      'self_comm': {'text': {'type': 'state', 'data': robot_self_comm_text, 'mask': 1}},
                      'query': {'question': {'type': 'state', 'data': robot_query, 'mask': 1}},
                      'reward': {'type': 'state', 'data': robot_reward},
                      'done': {'type': 'state', 'data': robot_done}}

        human_info = {'agent_obs': {'img': {'type': 'image', 'data': human_img_obs, 'mask': 1},
                                    'img2': {'type': 'image', 'data': human_img_obs2, 'mask': 1},
                                    'pos': {'type': 'state', 'data': self.human_pos.copy(), 'mask': 1}},
                      'other_comm': {'text': {'type': 'state', 'data': robot_self_comm_text, 'mask': float(random.randrange(0, 100) < 100 * self.comm_rate)}},
                      'self_comm': {'text': {'type': 'state', 'data': human_self_comm_text, 'mask': 1}},
                      'query': {'question': {'type': 'state', 'data': human_query, 'mask': 1}},
                      'reward': {'type': 'state', 'data': human_reward},
                      'done': {'type': 'state', 'data': human_done}}

        # get perspective invariant information
        pi_obs = self.instruction_imgs[self.state['instruction']]
        pi_obs2 = self.pi_obs2_imgs[self.state['fetched_item']]

        pi_info = {'img': {'type': 'image', 'data': pi_obs, 'mask': 1},
                   'img2': {'type': 'image', 'data': pi_obs2, 'mask': 1} }
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
            robot_other_comm, _ = self.create_text_query(self.state['instruction'], query=robot_query)
            robot_info['other_comm']['text']['data'] = np.concatenate([robot_other_comm, self.human_pos], axis=-1)
            robot_info['other_comm']['text']['mask'] = 1.0 if robot_query[0] != 0 else 0.0
            human_other_comm, _ = self.create_text_query(self.state['instruction'], query=human_query)
            human_info['other_comm']['text']['data'] = np.concatenate([human_other_comm, self.robot_pos], axis=-1)
            human_info['other_comm']['text']['mask'] = 1.0 if human_query[0] != 0 else 0.0

        self.robot_img_buffer = np.concatenate([self.robot_img_buffer, robot_img_obs[np.newaxis,...]], axis=0)
        self.robot_img2_buffer = np.concatenate([self.robot_img2_buffer, robot_img_obs2[np.newaxis,...]], axis=0)
        self.robot_pos_buffer = np.concatenate([self.robot_pos_buffer, self.robot_pos.copy()[np.newaxis,...]], axis=0)
        self.robot_action_buffer = np.concatenate([self.robot_action_buffer, robot_action[np.newaxis,...]], axis=0)
        self.robot_other_comm_data_buffer = np.concatenate([self.robot_other_comm_data_buffer,
                                                            robot_info['other_comm']['text']['data'][np.newaxis,...]], axis=0)
        self.robot_other_comm_mask_buffer = np.concatenate([self.robot_other_comm_mask_buffer,
                                                            np.array([robot_info['other_comm']['text']['mask']])[np.newaxis,...]], axis=0)

        self.human_img_buffer = np.concatenate([self.human_img_buffer, human_img_obs[np.newaxis,...]], axis=0)
        self.human_img2_buffer = np.concatenate([human_img_obs2[np.newaxis,...]], axis=0)
        self.human_pos_buffer = np.concatenate([self.human_pos.copy()[np.newaxis,...]], axis=0)
        self.human_action_buffer = np.concatenate([human_action[np.newaxis,...]], axis=0)
        self.human_other_comm_data_buffer = np.concatenate([human_info['other_comm']['text']['data'][np.newaxis,...]], axis=0)
        self.human_other_comm_mask_buffer = np.concatenate([np.array([human_info['other_comm']['text']['mask']])[np.newaxis,...]], axis=0)

        self.pi_img_buffer = np.concatenate([pi_obs[np.newaxis,...]], axis=0)
        self.pi_img2_buffer = np.concatenate([pi_obs2[np.newaxis,...]], axis=0)

        self.update_robot_belief(query=robot_query)

        self.robot_total_reward += robot_reward
        self.human_total_reward += human_reward
        if robot_reward > 0.0:
            self.success = 1.0
        else:
            self.success = 0.0
        self.time += 1.0

        return robot_info, human_info, pi_info, env_info

    def get_reward_done(self, action, agent_pos):
        # if agent_type == 'human':
        if agent_pos == 1:
            reward = -0.2
            bonus = 1.0
            penalty = -3.0
            done = 0.0
            action_ind = np.where(action == 1)[0][0]
            if self.state['fetched_item'] == self.state['instruction']:
                if action[7] == 1:
                    reward = bonus
                    done = 1.0
            elif action[7] == 1 and (self.state['fetched_item'] != self.state['instruction']):
                reward = penalty
                done = 1.0
            if action_ind == 8:
                reward = -2
            return reward, done
        else:
            reward = -0.2
            bonus = 1.0
            penalty = -3.0
            done = 0.0
            action_ind = np.where(action == 1)[0][0]
            if self.state['fetched_item'] is not '99':
                if (self.state['instruction'] == '00' and action_ind == 1) or (self.state['instruction'] == '01' and action_ind == 2) or \
                        (self.state['instruction'] == '10' and action_ind == 3) or (self.state['instruction'] == '11' and action_ind == 4) or \
                        (self.state['instruction'] == '22' and action_ind == 5) or (self.state['instruction'] == '23' and action_ind == 6):
                    reward = bonus
                elif 1 <= action_ind <= 6:
                    reward = penalty
            # penalise switching position
            if action_ind == 8:  # 10:
                reward = -2
        return reward, done

    def sample_action(self, idle=False):
        # idle, fetch tool 00, fetch tool 01, fetch tool 02, fetch tool 10, fetch tool 11, fetch tool 21,
        # assemble 0, assemble 1, assemble 2
        if idle:
            return self.action_list[0].copy()
        else:
            if self.switch_action:
                return random.choice(self.action_list)
            else:  # last action is the switch action
                return random.choice(self.action_list[:-1])

    def get_phy_actions(self, device, agent='robot', default=False):
        action_dicts = []
        for action_np in self.action_list:
            action_data = torch.as_tensor(action_np).to(device).float()
            action_dict = {'type': 'phy', 'agent': agent, 'data': action_data, 'target_agent': 'env'}
            action_dicts += [action_dict]
        return action_dicts

    def get_human_feedback(self, ):
        return None

    def get_metric(self):
        return {'robot_total_reward': self.robot_total_reward,
                'human_total_reward': self.human_total_reward,
                'success': self.success,
                'time': self.time}

    def human_policy(self):
        human_phy_action = np.zeros(self.action_size)
        if self.human_pos == 1:
            if self.state['fetched_item'] == self.state['instruction']:
                human_phy_action[7] = 1
            else:
                human_phy_action[0] = 1
        if self.human_pos == 0:
            human_phy_action = self.sample_action(idle=True)
        return human_phy_action

    def create_text_query(self, instruction_key='99', query='None'):
        #QUERY: [nothing, specific, type, comm]
        comm_text = np.zeros(14)
        if instruction_key == '99':
            comm_ind = 0
            query = self.query_list[0]
        else:
            if query == 'None':
                query_val = np.random.choice((1, 2, 3), p =(1/3, 1/3, 1/3))
            else:
                query_val = np.where(query == 1)[0][0]
            if query_val == 1:
                comm_ind = self.specific_comm_dict[instruction_key]
                query = self.query_list[1]
            elif query_val == 2:
                comm_ind = self.type_comm_dict[instruction_key]
                query = self.query_list[2]
            elif query_val == 3:
                comm_ind = self.color_comm_dict[instruction_key]
                query = self.query_list[3]
            else: #query val = 0
                comm_ind = 0
                query = self.query_list[0]
        comm_text[comm_ind] = 1
        return comm_text, query
    def update_robot_belief(self, query):
        '''
        robot doesn't see instr, only fetched tool and comms
        QUERY: [nothing, specific, type, comm]

        '''
        if query == 1: #specific
            self.robot_instruction_belief = self.state['instruction']
        elif query==2: #type
            if self.state['instruction'] == '00' or self.state['instruction'] == '01' :
                self.robot_instruction_belief = [instr for instr in self.robot_instruction_belief if instr in ('00','01') ]
            elif self.state['instruction'] == '10' or self.state['instruction'] == '11':
                self.robot_instruction_belief = [instr for instr in self.robot_instruction_belief if instr in ('10','11')]
            elif self.state['instruction'] == '22' or self.state['instruction'] == '23':
                self.robot_instruction_belief = [instr for instr in self.robot_instruction_belief if instr in ['22','23']]
        elif query==3: #colour
            if self.state['instruction'] == '00' or self.state['instruction'] == '10':
                self.robot_instruction_belief = [instr for instr in self.robot_instruction_belief if instr in ['00','10']]
            elif self.state['instruction'] == '01' or self.state['instruction'] == '11':
                self.robot_instruction_belief = [instr for instr in self.robot_instruction_belief if instr in ['01','11']]
            elif self.state['instruction'] == '22':
                self.robot_instruction_belief = [instr for instr in self.robot_instruction_belief if instr in ['22']]
            elif self.state['instruction'] == '23':
                self.robot_instruction_belief = [instr for instr in self.robot_instruction_belief if instr in ['23']]

        proba_val = 1 / len(self.robot_instruction_belief)
        fetched_item_ind = self.robot_obs_keys.index(self.state['fetched_item'])
        for instr in self.robot_instruction_belief:
            ind = self.instruction_keys.index(instr)
            self.robot_belief_probas[fetched_item_ind][ind] = proba_val

        return self.robot_belief_probas

