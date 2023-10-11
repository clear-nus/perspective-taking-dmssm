import utils.tools as tools


def fetch_tool_unity_config_hvae_ccat():
    config = tools.AttrDict()
    # General.
    config.seed = 501
    config.model_id = True
    config.rand_act = True
    config.method = 'hvae_transformer_cat'
    config.test = False

    config.h_size = 24
    config.num_cat = 2
    config.s_size = 48
    config.z_size = 32
    config.zp_size = 2

    config.comm_action_size = 6
    config.query_action_size = 4

    config.agent_rec_scale = {'img': 1000, 'img2': 1000, 'pos': 10}
    config.agent_embed_rec_scale = 1
    config.agent_rec_lb = {'img': 5, 'img2': 5, 'pos': 0}
    config.pi_rec_scale = {'img': 2000, 'img2': 2000}
    config.xp_rec_scale = 20
    config.pi_embed_rec_scale = 1
    config.pi_rec_lb = {'img': 5, 'img2': 5}
    config.pt_rec_scale = 1
    config.reward_rec_scale = 1
    config.kldiv_s_scale = 1
    config.kldiv_h_scale = 1
    config.kldiv_zp_scale = 1
    config.anneal_factor = 5.0
    config.anneal_factor_end = 0.3
    config.anneal_factor_decay = 0.9

    # Training.
    config.batch_length = 8
    config.batch_size = 64
    config.n_games = 700
    config.train_every = 25
    config.train_steps = 100
    config.lr = 1e-4
    config.pt_lr = 1e-4
    config.human_policy_lr = 1e-5
    config.max_buffer_size = 10000
    config.max_episode_length = 10

    config.discount = 0.1

    # Testing
    config.n_test_episode = 30
    return config


def fetch_tool_unity_config_deter_rnn():
    config = tools.AttrDict()
    # General.
    config.seed = 500
    config.model_id = True
    config.rand_act = True
    config.method = 'deter_rnn'
    config.test = False

    config.s_size = 32
    config.num_cat = 2
    config.e_size = 8
    config.z_size = 32
    config.zp_size = 2

    config.comm_action_size = 6
    config.query_action_size = 4

    config.agent_rec_scale = {'img': 1000, 'img2': 1000, 'pos': 10}
    config.agent_embed_rec_scale = 1
    config.agent_rec_lb = {'img': 5, 'img2': 5, 'pos': 0}
    config.pi_rec_scale = {'img': 2000, 'img2': 2000}
    config.xp_rec_scale = 20
    config.pi_embed_rec_scale = 1
    config.pi_rec_lb = {'img': 5, 'img2': 5}
    config.pt_rec_scale = 1
    config.reward_rec_scale = 1
    config.anneal_factor = 5.0
    config.anneal_factor_end = 0.3
    config.anneal_factor_decay = 0.9

    # Training
    config.batch_length = 8
    config.batch_size = 64
    config.n_games = 700
    config.train_every = 25
    config.train_steps = 100
    config.lr = 1e-4
    config.pt_lr = 1e-4
    config.human_policy_lr = 1e-5
    config.max_buffer_size = 10000
    config.max_episode_length = 10

    config.discount = 0.9

    # Testing
    config.n_test_episode = 30
    return config

def fetch_tool_unity_config_vae():
    config = tools.AttrDict()
    # General.
    config.seed = 500
    config.model_id = True
    config.rand_act = True
    config.method = 'vae'
    config.test = False

    config.s_size = 32
    config.num_cat = 2
    config.e_size = 8
    config.z_size = 32
    config.zp_size = 2

    config.comm_action_size = 6
    config.query_action_size = 4

    config.agent_rec_scale = {'img': 1000, 'img2': 1000, 'pos': 10}
    config.agent_embed_rec_scale = 1
    config.agent_rec_lb = {'img': 5, 'img2': 5, 'pos': 0}
    config.pi_rec_scale = {'img': 2000, 'img2': 2000}
    config.xp_rec_scale = 20
    config.pi_embed_rec_scale = 1
    config.pi_rec_lb = {'img': 5, 'img2': 5}
    config.pt_rec_scale = 1
    config.reward_rec_scale = 1
    config.anneal_factor = 5.0
    config.anneal_factor_end = 0.3
    config.anneal_factor_decay = 0.9
    config.kldiv_s_scale = 1

    # Training
    config.batch_length = 8
    config.batch_size = 64
    config.n_games = 700
    config.train_every = 25
    config.train_steps = 100
    config.lr = 1e-4
    config.pt_lr = 1e-4
    config.human_policy_lr = 1e-5
    config.max_buffer_size = 10000
    config.max_episode_length = 10

    config.discount = 0.9

    # Testing
    config.n_test_episode = 30
    return config

# def fetch_tool_config_liam():
#     config = tools.AttrDict()
#     # General.
#     config.seed = 500
#     config.model_id = True
#     config.rand_act = True
#     config.method = 'liam'
#     config.test = False
#
#     config.z_size = 32
#
#     config.comm_action_size = 6
#     config.query_action_size = 4
#
#     config.obs_rec_scale = {'img': 1000, 'img2': 1000, 'pos': 10}
#     config.obs_embed_rec_scale = 1
#     config.obs_rec_lb = {'img': 10, 'img2': 10, 'pos': 0}
#     config.action_rec_scale = 50
#     config.pi_embed_rec_scale = 1
#     config.pi_rec_lb = {'img': 0}
#     config.pt_rec_scale = 1
#     config.reward_rec_scale = 1
#
#     # Training
#     config.batch_length = 4
#     config.batch_size = 64
#     config.n_games = 200
#     config.train_every = 25
#     config.train_steps = 100
#     config.lr = 1e-4
#     config.om_lr = 5e-4
#     config.rl_lr = 5e-4
#     config.human_policy_lr = 1e-5
#     config.max_buffer_size = 10000
#     config.max_episode_length = 10
#
#     config.gamma = 0.99
#     config.discount = 0.9
#     config.gae_lambda = 0.95
#     config.entropy_coef = 0.2
#
#     # Testing
#     config.n_test_episode = 30
#     return config
