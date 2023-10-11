import utils.tools as tools


def table_assembly_config_hvae_ccat():
    config = tools.AttrDict()
    # General.
    config.seed = 500
    config.model_id = True
    config.rand_act = True
    config.method = 'hvae_transformer_cat'
    config.test = False

    config.h_size = 64
    config.num_cat = 8
    config.s_size = 64
    config.e_size = 16
    config.z_size = 32
    config.zp_size = 16

    config.comm_action_size = 9
    config.query_action_size = 2

    config.agent_rec_scale = {'img': 12288, 'pos': 300}
    config.agent_embed_rec_scale = 64
    config.agent_rec_lb = {'img': 0, 'pos': 0}
    config.pi_rec_scale = {'img': 12288}
    config.pi_embed_rec_scale = 64
    config.pi_rec_lb = {'img': 0}
    config.xp_rec_scale = 300
    config.pt_rec_scale = 64
    config.reward_rec_scale = 100

    config.kldiv_s_scale = 1.0
    config.kldiv_h_scale = 1.0
    config.kldiv_zp_scale = 1.0
    config.anneal_factor = 3.0
    config.anneal_factor_end = 1.0
    config.anneal_factor_decay = 0.9

    config.kl_dyn_s_free_bits = 0.0
    config.kl_rep_s_free_bits = 1.0
    config.kl_dyn_h_free_bits = 0.0
    config.kl_rep_h_free_bits = 1.0
    config.kl_dyn_zp_free_bits = 0.0
    config.kl_rep_zp_free_bits = 1.0

    config.kl_dyn_s_scale = 0.5
    config.kl_rep_s_scale = 0.1
    config.kl_dyn_h_scale = 0.5
    config.kl_rep_h_scale = 0.1
    config.kl_dyn_zp_scale = 0.5
    config.kl_rep_zp_scale = 0.1

    # Training.
    config.batch_length = 4
    config.batch_size = 64
    config.n_games = 2000
    config.train_every = 25
    config.train_steps = 100
    config.lr = 1e-4
    config.pt_lr = 1e-4
    config.human_policy_lr = 1e-4
    config.max_buffer_size = 10000
    config.max_episode_length = 10

    config.discount = 0.9

    # Testing
    config.n_test_episode = 100
    return config


def table_assembly_config_deter_rnn():
    config = tools.AttrDict()
    # General.
    config.seed = 500
    config.model_id = True
    config.rand_act = True
    config.method = 'hvae_transformer_cat'
    config.test = False

    config.num_cat = 8
    config.s_size = 128
    config.e_size = 16
    config.z_size = 32
    config.zp_size = 16

    config.comm_action_size = 9
    config.query_action_size = 2

    config.agent_rec_scale = {'img': 12288, 'pos': 300}
    config.agent_embed_rec_scale = 64
    config.agent_rec_lb = {'img': 0, 'pos': 0}
    config.pi_rec_scale = {'img': 12288}
    config.pi_embed_rec_scale = 64
    config.pi_rec_lb = {'img': 0}
    config.xp_rec_scale = 300
    config.pt_rec_scale = 64
    config.reward_rec_scale = 100

    config.kldiv_s_scale = 1.0
    config.kldiv_h_scale = 1.0
    config.kldiv_zp_scale = 1.0
    config.anneal_factor = 3.0
    config.anneal_factor_end = 1.0
    config.anneal_factor_decay = 0.9

    config.kl_dyn_s_free_bits = 0.0
    config.kl_rep_s_free_bits = 1.0
    config.kl_dyn_h_free_bits = 0.0
    config.kl_rep_h_free_bits = 1.0
    config.kl_dyn_zp_free_bits = 0.0
    config.kl_rep_zp_free_bits = 1.0

    config.kl_dyn_s_scale = 0.5
    config.kl_rep_s_scale = 0.1
    config.kl_dyn_h_scale = 0.5
    config.kl_rep_h_scale = 0.1
    config.kl_dyn_zp_scale = 0.5
    config.kl_rep_zp_scale = 0.1

    # Training.
    config.batch_length = 4
    config.batch_size = 64
    config.n_games = 2000
    config.train_every = 25
    config.train_steps = 100
    config.lr = 1e-4
    config.pt_lr = 1e-4
    config.human_policy_lr = 1e-4
    config.max_buffer_size = 10000
    config.max_episode_length = 10

    config.discount = 0.9

    # Testing
    config.n_test_episode = 100
    return config

def table_assembly_config_vae():
    config = tools.AttrDict()
    # General.
    config.seed = 500
    config.model_id = True
    config.rand_act = True
    config.method = 'hvae_transformer_cat'
    config.test = False

    config.num_cat = 8
    config.s_size = 128
    config.e_size = 16
    config.z_size = 32
    config.zp_size = 16

    config.comm_action_size = 9
    config.query_action_size = 2

    config.agent_rec_scale = {'img': 12288, 'pos': 300}
    config.agent_embed_rec_scale = 64
    config.agent_rec_lb = {'img': 0, 'pos': 0}
    config.pi_rec_scale = {'img': 12288}
    config.pi_embed_rec_scale = 64
    config.pi_rec_lb = {'img': 0}
    config.xp_rec_scale = 300
    config.pt_rec_scale = 64
    config.reward_rec_scale = 100

    config.kldiv_s_scale = 1.0
    config.kldiv_h_scale = 1.0
    config.kldiv_zp_scale = 1.0
    config.anneal_factor = 3.0
    config.anneal_factor_end = 1.0
    config.anneal_factor_decay = 0.9

    config.kl_dyn_s_free_bits = 0.0
    config.kl_rep_s_free_bits = 1.0
    config.kl_dyn_h_free_bits = 0.0
    config.kl_rep_h_free_bits = 1.0
    config.kl_dyn_zp_free_bits = 0.0
    config.kl_rep_zp_free_bits = 1.0

    config.kl_dyn_s_scale = 0.5
    config.kl_rep_s_scale = 0.1
    config.kl_dyn_h_scale = 0.5
    config.kl_rep_h_scale = 0.1
    config.kl_dyn_zp_scale = 0.5
    config.kl_rep_zp_scale = 0.1

    # Training.
    config.batch_length = 4
    config.batch_size = 64
    config.n_games = 2000
    config.train_every = 25
    config.train_steps = 100
    config.lr = 1e-4
    config.pt_lr = 1e-4
    config.human_policy_lr = 1e-4
    config.max_buffer_size = 10000
    config.max_episode_length = 10

    config.discount = 0.9

    # Testing
    config.n_test_episode = 100
    return config
def table_assembly_config_liam():
    config = tools.AttrDict()
    # General.
    config.seed = 500
    config.model_id = True
    config.rand_act = True
    config.method = 'liam'
    config.test = False

    config.z_size = 32

    config.comm_action_size = 6
    config.query_action_size = 4

    config.obs_rec_scale = {'img': 1000, 'pos': 10}
    config.obs_embed_rec_scale = 1
    config.obs_rec_lb = {'img': 10, 'pos': 0}
    config.action_rec_scale = 50
    config.pi_embed_rec_scale = 1
    config.pi_rec_lb = {'img': 0}
    config.pt_rec_scale = 1
    config.reward_rec_scale = 1

    # Training
    config.batch_length = 4
    config.batch_size = 64
    config.n_games = 1000
    config.train_every = 25
    config.train_steps = 100
    config.lr = 1e-4
    config.om_lr = 5e-4
    config.rl_lr = 5e-4
    config.human_policy_lr = 1e-5
    config.max_buffer_size = 10000
    config.max_episode_length = 10

    config.gamma = 0.99
    config.discount = 0.9
    config.gae_lambda = 0.95
    config.entropy_coef = 0.2

    # Testing
    config.n_test_episode = 100
    return config
