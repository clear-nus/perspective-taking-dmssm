import numpy as np


class ReplayBufferModelBased:
    def __init__(self, max_size, agent_obs_info_dict, agent_comm_info_dict, query_info_dict, pi_info_dict, action_shape, pos_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.agent_obs_info_dict = agent_obs_info_dict
        self.other_comm_info_dict = agent_comm_info_dict
        self.self_comm_info_dict = agent_comm_info_dict
        self.query_info_dict = query_info_dict
        self.pi_info_dict = pi_info_dict

        self.agent_obs_dict = {}
        self.agent_obs_mask_dict = {}
        for obs_name, obs_info in agent_obs_info_dict.items():
            self.agent_obs_dict[obs_name] = np.zeros((self.mem_size, *obs_info['modality_size']))
            self.agent_obs_mask_dict[obs_name] = np.zeros((self.mem_size, 1))

        self.other_comm_dict = {}
        self.other_comm_mask_dict = {}
        for comm_name, comm_info in agent_comm_info_dict.items():
            self.other_comm_dict[comm_name] = np.zeros((self.mem_size, *comm_info['modality_size']))
            self.other_comm_mask_dict[comm_name] = np.zeros((self.mem_size, 1))

        self.self_comm_dict = {}
        self.self_comm_mask_dict = {}
        for comm_name, comm_info in agent_comm_info_dict.items():
            self.self_comm_dict[comm_name] = np.zeros((self.mem_size, *comm_info['modality_size']))
            self.self_comm_mask_dict[comm_name] = np.zeros((self.mem_size, 1))

        self.query_dict = {}
        self.query_mask_dict = {}  # not useful
        for query_name, query_info in query_info_dict.items():
            self.query_dict[query_name] = np.zeros((self.mem_size, *query_info['modality_size']))
            self.query_mask_dict[query_name] = np.zeros((self.mem_size, 1))

        self.pi_dict = {}
        self.pi_mask_dict = {}
        for pi_name, pi_info in pi_info_dict.items():
            self.pi_dict[pi_name] = np.zeros((self.mem_size, *pi_info['modality_size']))
            self.pi_mask_dict[pi_name] = np.zeros((self.mem_size, 1))

        self.self_action = np.zeros((self.mem_size, *action_shape))
        self.other_action = np.zeros((self.mem_size, *action_shape))
        self.reward = np.zeros(self.mem_size)
        self.terminal = np.zeros(self.mem_size, dtype=np.bool)
        self.pos = np.zeros((self.mem_size, *pos_shape))

    def store_transition(self, agent_obs_dict, other_comm_dict, self_comm_dict, query_dict, pi_obs_dict, pos, self_action, other_action, reward, done):
        index = self.mem_cntr % self.mem_size

        for obs_name, obs_buffer in self.agent_obs_dict.items():
            obs_buffer[index] = agent_obs_dict[obs_name]['data']

        for obs_name, obs_mask_buffer in self.agent_obs_mask_dict.items():
            obs_mask_buffer[index] = agent_obs_dict[obs_name]['mask']

        for comm_name, comm_buffer in self.other_comm_dict.items():
            comm_buffer[index] = other_comm_dict[comm_name]['data']

        for comm_name, comm_mask_buffer in self.other_comm_mask_dict.items():
            comm_mask_buffer[index] = other_comm_dict[comm_name]['mask']

        for comm_name, comm_buffer in self.self_comm_dict.items():
            comm_buffer[index] = self_comm_dict[comm_name]['data']

        for comm_name, comm_mask_buffer in self.self_comm_mask_dict.items():
            comm_mask_buffer[index] = self_comm_dict[comm_name]['mask']

        for query_name, query_buffer in self.query_dict.items():
            query_buffer[index] = query_dict[query_name]['data']

        for query_name, query_mask_buffer in self.query_mask_dict.items():
            query_mask_buffer[index] = query_dict[query_name]['mask']

        for pi_name, pi_buffer in self.pi_dict.items():
            pi_buffer[index] = pi_obs_dict[pi_name]['data']

        for pi_name, pi_mask_buffer in self.pi_mask_dict.items():
            pi_mask_buffer[index] = pi_obs_dict[pi_name]['mask']

        self.self_action[index] = self_action['data']
        self.other_action[index] = other_action['data']
        self.reward[index] = reward['data']
        self.terminal[index] = done['data']
        self.pos[index] = pos['data']

        self.mem_cntr += 1

    def sample_sequence_buffer(self, time_size, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        # batch = np.random.choice(max_mem, batch_size)
        batch_id = np.random.choice(max_mem - time_size * batch_size, 1).item()
        batch = [i for i in range(batch_id, batch_id + time_size * batch_size)]

        agent_obs_dict = {}
        for obs_name, obs_buffer in self.agent_obs_dict.items():
            if self.agent_obs_info_dict[obs_name]['modality_type'] == 'image':
                agent_obs_dict[obs_name] = {'type': 'image',
                                            'data': obs_buffer[batch].reshape(batch_size, time_size, 64, 64, 3).transpose((1, 0, 4, 2, 3)),
                                            'mask': self.agent_obs_mask_dict[obs_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}

            elif self.agent_obs_info_dict[obs_name]['modality_type'] == 'state':
                agent_obs_dict[obs_name] = {'type': 'state',
                                            'data': obs_buffer[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1)),
                                            'mask': self.agent_obs_mask_dict[obs_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}
            else:
                agent_obs_dict[obs_name] = {'type': 'state',
                                            'data': obs_buffer[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1)),
                                            'mask': self.agent_obs_mask_dict[obs_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}

        other_comm_dict = {}
        for comm_name, comm_buffer in self.other_comm_dict.items():
            if self.other_comm_info_dict[comm_name]['modality_type'] == 'image':
                other_comm_dict[comm_name] = {'type': 'image',
                                              'data': comm_buffer[batch].reshape(batch_size, time_size, 64, 64, 3).transpose((1, 0, 4, 2, 3)),
                                              'mask': self.other_comm_mask_dict[comm_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}

            elif self.other_comm_info_dict[comm_name]['modality_type'] == 'state':
                other_comm_dict[comm_name] = {'type': 'state',
                                              'data': comm_buffer[batch].reshape(batch_size, time_size, -1).transpose(
                                                  (1, 0, -1)),
                                              'mask': self.other_comm_mask_dict[comm_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}
            else:
                other_comm_dict[comm_name] = {'type': 'state',
                                              'data': comm_buffer[batch].reshape(batch_size, time_size, -1).transpose(
                                                  (1, 0, -1)),
                                              'mask': self.other_comm_mask_dict[comm_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}

        self_comm_dict = {}
        for comm_name, comm_buffer in self.self_comm_dict.items():
            if self.self_comm_info_dict[comm_name]['modality_type'] == 'image':
                self_comm_dict[comm_name] = {'type': 'image',
                                             'data': comm_buffer[batch].reshape(batch_size, time_size, 64, 64,
                                                                                3).transpose(
                                                 (1, 0, 4, 2, 3)),
                                             'mask': self.self_comm_mask_dict[comm_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}
            elif self.self_comm_info_dict[comm_name]['modality_type'] == 'state':
                self_comm_dict[comm_name] = {'type': 'state',
                                             'data': comm_buffer[batch].reshape(batch_size, time_size, -1).transpose(
                                                 (1, 0, -1)),
                                             'mask': self.self_comm_mask_dict[comm_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}
            else:
                self_comm_dict[comm_name] = {'type': 'state',
                                             'data': comm_buffer[batch].reshape(batch_size, time_size, -1).transpose(
                                                 (1, 0, -1)),
                                             'mask': self.self_comm_mask_dict[comm_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}

        query_dict = {}
        for query_name, query_buffer in self.query_dict.items():
            if self.query_info_dict[query_name]['modality_type'] == 'image':
                query_dict[query_name] = {'type': 'image',
                                          'data': query_buffer[batch].reshape(batch_size, time_size, 64, 64,
                                                                              3).transpose(
                                              (1, 0, 4, 2, 3)),
                                          'mask': self.query_mask_dict[query_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}
            elif self.query_info_dict[query_name]['modality_type'] == 'state':
                query_dict[query_name] = {'type': 'state',
                                          'data': query_buffer[batch].reshape(batch_size, time_size, -1).transpose(
                                              (1, 0, -1)),
                                          'mask': self.query_mask_dict[query_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}
            else:
                query_dict[query_name] = {'type': 'state',
                                          'data': query_buffer[batch].reshape(batch_size, time_size, -1).transpose(
                                              (1, 0, -1)),
                                          'mask': self.query_mask_dict[query_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}

        pi_dict = {}
        for pi_name, pi_buffer in self.pi_dict.items():
            if self.pi_info_dict[pi_name]['modality_type'] == 'image':
                pi_dict[pi_name] = {'type': 'image',
                                    'data': pi_buffer[batch].reshape(batch_size, time_size, 64, 64, 3).transpose(
                                        (1, 0, 4, 2, 3)),
                                    'mask': self.pi_mask_dict[pi_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}
            elif self.pi_info_dict[pi_name]['modality_type'] == 'state':
                pi_dict[pi_name] = {'type': 'state',
                                    'data': pi_buffer[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1)),
                                    'mask': self.pi_mask_dict[pi_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}
            else:
                pi_dict[pi_name] = {'type': 'image',
                                    'data': pi_buffer[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1)),
                                    'mask': self.pi_mask_dict[pi_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}

        self_action = self.self_action[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))
        other_action = self.other_action[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))
        reward = self.reward[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))
        dones = self.terminal[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))
        pos = self.pos[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))

        # import cv2
        # visualize_images = self.robot_obs[batch].reshape(batch_size, time_size, 64, 64, 3).transpose((1, 0, 2, 3, 4))
        # for i in range(time_size):
        #     cv2.imshow('test', visualize_images[i, 0])
        #     cv2.waitKey(200)
        #     print(i)
        return agent_obs_dict, other_comm_dict, self_comm_dict, query_dict, pi_dict, pos, self_action, other_action, reward, dones


class ReplayBufferModelFree:
    def __init__(self, max_size, agent_obs_info_dict, agent_comm_info_dict, query_info_dict, pi_info_dict,
                 action_shape, comm_shape, feedback_shape, query_shape,
                 pos_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.agent_obs_info_dict = agent_obs_info_dict
        self.other_comm_info_dict = agent_comm_info_dict
        self.self_comm_info_dict = agent_comm_info_dict
        self.query_info_dict = query_info_dict
        self.pi_info_dict = pi_info_dict

        self.agent_obs_dict = {}
        self.agent_obs_mask_dict = {}
        for obs_name, obs_info in agent_obs_info_dict.items():
            self.agent_obs_dict[obs_name] = np.zeros((self.mem_size, *obs_info['modality_size']))
            self.agent_obs_mask_dict[obs_name] = np.zeros((self.mem_size, 1))

        self.other_comm_dict = {}
        self.other_comm_mask_dict = {}
        for comm_name, comm_info in agent_comm_info_dict.items():
            self.other_comm_dict[comm_name] = np.zeros((self.mem_size, *comm_info['modality_size']))
            self.other_comm_mask_dict[comm_name] = np.zeros((self.mem_size, 1))

        self.self_comm_dict = {}
        self.self_comm_mask_dict = {}
        for comm_name, comm_info in agent_comm_info_dict.items():
            self.self_comm_dict[comm_name] = np.zeros((self.mem_size, *comm_info['modality_size']))
            self.self_comm_mask_dict[comm_name] = np.zeros((self.mem_size, 1))

        self.query_dict = {}
        self.query_mask_dict = {}  # not useful
        for query_name, query_info in query_info_dict.items():
            self.query_dict[query_name] = np.zeros((self.mem_size, *query_info['modality_size']))
            self.query_mask_dict[query_name] = np.zeros((self.mem_size, 1))

        self.pi_dict = {}
        self.pi_mask_dict = {}
        for pi_name, pi_info in pi_info_dict.items():
            self.pi_dict[pi_name] = np.zeros((self.mem_size, *pi_info['modality_size']))
            self.pi_mask_dict[pi_name] = np.zeros((self.mem_size, 1))

        self.self_action = np.zeros((self.mem_size, *action_shape))
        self.self_comm_action = np.zeros((self.mem_size, *comm_shape))
        self.self_query_action = np.zeros((self.mem_size, *query_shape))
        self.other_action = np.zeros((self.mem_size, *action_shape))
        self.reward = np.zeros(self.mem_size)
        self.terminal = np.zeros(self.mem_size, dtype=np.bool)
        self.pos = np.zeros((self.mem_size, *pos_shape))

    def store_transition(self, agent_obs_dict, other_comm_dict, self_comm_dict, query_dict, pi_obs_dict, pos,
                         self_action, self_comm_action, self_query_action, other_action, reward, done):
        index = self.mem_cntr % self.mem_size

        for obs_name, obs_buffer in self.agent_obs_dict.items():
            obs_buffer[index] = agent_obs_dict[obs_name]['data']

        for obs_name, obs_mask_buffer in self.agent_obs_mask_dict.items():
            obs_mask_buffer[index] = agent_obs_dict[obs_name]['mask']

        for comm_name, comm_buffer in self.other_comm_dict.items():
            comm_buffer[index] = other_comm_dict[comm_name]['data']

        for comm_name, comm_mask_buffer in self.other_comm_mask_dict.items():
            comm_mask_buffer[index] = other_comm_dict[comm_name]['mask']

        for comm_name, comm_buffer in self.self_comm_dict.items():
            comm_buffer[index] = self_comm_dict[comm_name]['data']

        for comm_name, comm_mask_buffer in self.self_comm_mask_dict.items():
            comm_mask_buffer[index] = self_comm_dict[comm_name]['mask']

        for query_name, query_buffer in self.query_dict.items():
            query_buffer[index] = query_dict[query_name]['data']

        for query_name, query_mask_buffer in self.query_mask_dict.items():
            query_mask_buffer[index] = query_dict[query_name]['mask']

        for pi_name, pi_buffer in self.pi_dict.items():
            pi_buffer[index] = pi_obs_dict[pi_name]['data']

        for pi_name, pi_mask_buffer in self.pi_mask_dict.items():
            pi_mask_buffer[index] = pi_obs_dict[pi_name]['mask']

        self.self_action[index] = self_action['data']
        self.self_comm_action[index] = self_comm_action['data']
        self.self_query_action[index] = self_query_action['data']

        self.other_action[index] = other_action['data']
        self.reward[index] = reward['data']
        self.terminal[index] = done['data']
        self.pos[index] = pos['data']

        self.mem_cntr += 1

    def sample_sequence_buffer(self, time_size, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        # batch = np.random.choice(max_mem, batch_size)
        batch_id = np.random.choice(max_mem - time_size * batch_size, 1).item()
        batch = [i for i in range(batch_id, batch_id + time_size * batch_size)]

        agent_obs_dict = {}
        for obs_name, obs_buffer in self.agent_obs_dict.items():
            if self.agent_obs_info_dict[obs_name]['modality_type'] == 'image':
                agent_obs_dict[obs_name] = {'type': 'image',
                                            'data': obs_buffer[batch].reshape(batch_size, time_size, 64, 64, 3).transpose((1, 0, 4, 2, 3)),
                                            'mask': self.agent_obs_mask_dict[obs_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}

            elif self.agent_obs_info_dict[obs_name]['modality_type'] == 'state':
                agent_obs_dict[obs_name] = {'type': 'state',
                                            'data': obs_buffer[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1)),
                                            'mask': self.agent_obs_mask_dict[obs_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}
            else:
                agent_obs_dict[obs_name] = {'type': 'state',
                                            'data': obs_buffer[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1)),
                                            'mask': self.agent_obs_mask_dict[obs_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}

        other_comm_dict = {}
        for comm_name, comm_buffer in self.other_comm_dict.items():
            if self.other_comm_info_dict[comm_name]['modality_type'] == 'image':
                other_comm_dict[comm_name] = {'type': 'image',
                                              'data': comm_buffer[batch].reshape(batch_size, time_size, 64, 64, 3).transpose((1, 0, 4, 2, 3)),
                                              'mask': self.other_comm_mask_dict[comm_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}

            elif self.other_comm_info_dict[comm_name]['modality_type'] == 'state':
                other_comm_dict[comm_name] = {'type': 'state',
                                              'data': comm_buffer[batch].reshape(batch_size, time_size, -1).transpose(
                                                  (1, 0, -1)),
                                              'mask': self.other_comm_mask_dict[comm_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}
            else:
                other_comm_dict[comm_name] = {'type': 'state',
                                              'data': comm_buffer[batch].reshape(batch_size, time_size, -1).transpose(
                                                  (1, 0, -1)),
                                              'mask': self.other_comm_mask_dict[comm_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}

        self_comm_dict = {}
        for comm_name, comm_buffer in self.self_comm_dict.items():
            if self.self_comm_info_dict[comm_name]['modality_type'] == 'image':
                self_comm_dict[comm_name] = {'type': 'image',
                                             'data': comm_buffer[batch].reshape(batch_size, time_size, 64, 64,
                                                                                3).transpose(
                                                 (1, 0, 4, 2, 3)),
                                             'mask': self.self_comm_mask_dict[comm_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}
            elif self.self_comm_info_dict[comm_name]['modality_type'] == 'state':
                self_comm_dict[comm_name] = {'type': 'state',
                                             'data': comm_buffer[batch].reshape(batch_size, time_size, -1).transpose(
                                                 (1, 0, -1)),
                                             'mask': self.self_comm_mask_dict[comm_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}
            else:
                self_comm_dict[comm_name] = {'type': 'state',
                                             'data': comm_buffer[batch].reshape(batch_size, time_size, -1).transpose(
                                                 (1, 0, -1)),
                                             'mask': self.self_comm_mask_dict[comm_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}

        query_dict = {}
        for query_name, query_buffer in self.query_dict.items():
            if self.query_info_dict[query_name]['modality_type'] == 'image':
                query_dict[query_name] = {'type': 'image',
                                          'data': query_buffer[batch].reshape(batch_size, time_size, 64, 64,
                                                                              3).transpose(
                                              (1, 0, 4, 2, 3)),
                                          'mask': self.query_mask_dict[query_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}
            elif self.query_info_dict[query_name]['modality_type'] == 'state':
                query_dict[query_name] = {'type': 'state',
                                          'data': query_buffer[batch].reshape(batch_size, time_size, -1).transpose(
                                              (1, 0, -1)),
                                          'mask': self.query_mask_dict[query_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}
            else:
                query_dict[query_name] = {'type': 'state',
                                          'data': query_buffer[batch].reshape(batch_size, time_size, -1).transpose(
                                              (1, 0, -1)),
                                          'mask': self.query_mask_dict[query_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}

        pi_dict = {}
        for pi_name, pi_buffer in self.pi_dict.items():
            if self.pi_info_dict[pi_name]['modality_type'] == 'image':
                pi_dict[pi_name] = {'type': 'image',
                                    'data': pi_buffer[batch].reshape(batch_size, time_size, 64, 64, 3).transpose(
                                        (1, 0, 4, 2, 3)),
                                    'mask': self.pi_mask_dict[pi_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}
            elif self.pi_info_dict[pi_name]['modality_type'] == 'state':
                pi_dict[pi_name] = {'type': 'state',
                                    'data': pi_buffer[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1)),
                                    'mask': self.pi_mask_dict[pi_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}
            else:
                pi_dict[pi_name] = {'type': 'image',
                                    'data': pi_buffer[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1)),
                                    'mask': self.pi_mask_dict[pi_name][batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))}

        self_action = self.self_action[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))
        self_comm_action = self.self_comm_action[batch].reshape(batch_size, time_size, -1).transpose(1, 0, -1)
        self_query_action = self.self_query_action[batch].reshape(batch_size, time_size, -1).transpose(1, 0, -1)
        other_action = self.other_action[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))
        reward = self.reward[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))
        dones = self.terminal[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))
        pos = self.pos[batch].reshape(batch_size, time_size, -1).transpose((1, 0, -1))

        # import cv2
        # visualize_images = self.robot_obs[batch].reshape(batch_size, time_size, 64, 64, 3).transpose((1, 0, 2, 3, 4))
        # for i in range(time_size):
        #     cv2.imshow('test', visualize_images[i, 0])
        #     cv2.waitKey(200)
        #     print(i)
        return agent_obs_dict, other_comm_dict, self_comm_dict, query_dict, pi_dict, pos, \
            self_action, self_comm_action, self_query_action, other_action, reward, dones
