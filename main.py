# TODO 对于迷宫类任务，还可以绘制路线图(参考论文).

# antmaze-large-diverse-v1

import os
import argparse
from logger import setup_logger
import gym
import d4rl
from utils import set_seed, ReplayBuffer, wrap_env
import numpy as np
import pickle
from datetime import datetime
from algos.PLAS import train_PLAS
from algos.LAPO import train_LAPO
from algos.SPOT import train_SPOT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=[0])                           # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--device", type=str, default='cuda')            # Device
    parser.add_argument('--train_on_server', type=bool, default=False)   # Train on server or PC

    # Logging and Model parameters
    parser.add_argument('--log_dir', type=str, default='./results/')     # Logging directory
    parser.add_argument("--save_freq", type=int, default=int(5e4))       # How often it saves the model
    parser.add_argument("--eval_freq", type=int, default=int(1e3))       # How often (time steps) evaluate model.

    # Env parameters
    parser.add_argument("--env_name", default="hopper-expert-v2")      # Env name
    parser.add_argument("--dataset", type=str, default=None)             # Path to dataset if not d4rl env
    parser.add_argument("--wrap_env_state", type=bool, default=True)     # If normalize the env state.

    # Algos parameters
    parser.add_argument("--algo_name", default="LAPO")                   # Algorithm: PLAS/LAPO/SPOT.
    parser.add_argument('--vae_mode', type=str, default='train')		 # train or load VAE(PLAS/SPOT)
    parser.add_argument('--buffer_size', type=int, default=int(2e6))     # buffer size for actor & critic training
    parser.add_argument('--vae_iteration', type=int, default=int(2e5))	 # vae training iterations, 5e5
    parser.add_argument("--AC_iteration", type=int, default=int(3e5))    # actor and critic training iterations
    parser.add_argument('--vae_kl_weight', type=float, default=0.5)      # kl_weight coefficient in vae loss
    parser.add_argument('--max_latent_action', type=float, default=2.)   # max action of the latent policy
    parser.add_argument('--batch_size', type=int, default=512)	         # batch size
    parser.add_argument('--vae_lr', type=float, default=2e-4)            # vae training learning rate
    parser.add_argument('--actor_lr', type=float, default=2e-4)	         # policy learning rate
    parser.add_argument('--critic_lr', type=float, default=2e-4)	     # policy learning rate
    parser.add_argument('--tau', type=float, default=0.005)	             # soft update frequency
    parser.add_argument('--gamma', type=float, default=0.99)             # discount
    parser.add_argument('--lmbda', type=float, default=0.75)             # clipped double Q-learning in PLAS and LAPO
    parser.add_argument('--latent_dim_coff', type=int, default=2)        # times action_dim
    parser.add_argument('--policy_noise', type=float, default=0.2)       # policy noise
    parser.add_argument('--policy_freq', type=int, default=2)            # policy delay update frequency in TD3

    parser.add_argument('--SPOT_iwae', type=bool, default=True)          # ELBO or IWAE  in SPOT
    parser.add_argument('--SPOT_lmbda', type=float, default=0.2)         # ELBO or IWAE coefficient in SPOT
    parser.add_argument('--SPOT_num_samples', type=int, default=1)

    parser.add_argument('--kl_annealing', type=bool, default=False)
    parser.add_argument('--batch_normalization', type=bool, default=True)
    parser.add_argument('--batch_normalization_weight', type=float, default=0.2)

    args = parser.parse_args()
    if args.dataset is None:
        args.dataset = args.env_name

    # -------------------------------------- Setup Environment ----------------------------------------#
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # --------------------------------------- Load DRL Dataset -----------------------------------------#
    if args.env_name == args.dataset:
        dataset = d4rl.qlearning_dataset(env)  # Load d4rl dataset
    else:
        if args.dataset == 'hopper-medium-expert':
            dataset1 = d4rl.qlearning_dataset(gym.make('hopper-medium-v2'))
            dataset2 = d4rl.qlearning_dataset(gym.make('hopper-expert-v2'))
            dataset = {key: np.concatenate([dataset1[key], dataset2[key]]) for key in dataset1.keys()}
            print("Loaded data from hopper-medium-v2 and hopper-expert-v2")
        elif args.dataset == 'hopper-random-medium':
            dataset1 = d4rl.qlearning_dataset(gym.make('hopper-medium-v2'))
            dataset2 = d4rl.qlearning_dataset(gym.make('hopper-random-v2'))
            dataset = {key: np.concatenate([dataset1[key], dataset2[key]]) for key in dataset1.keys()}
            print("Loaded data from hopper-medium-v2 and hopper-random-v2")
        else:
            dataset_file = os.path.dirname(os.path.abspath(__file__)) + '/dataset/'+args.dataset + '.pkl'
            dataset = pickle.load(open(dataset_file, 'rb'))
            print("Loaded data from " + dataset_file)

    # --------------------------------------- Value in LAPO -----------------------------------------#
    if args.algo_name == 'LAPO':
        if 'antmaze' in args.env_name:
            dataset['rewards'] = (dataset['rewards']*100)
            min_v = 0
            max_v = 1 * 100
        else:
            dataset['rewards'] = dataset['rewards']/dataset['rewards'].max()
            min_v = dataset['rewards'].min()/(1-args.gamma)
            max_v = dataset['rewards'].max()/(1-args.gamma)

    # ----------------------------------- Replay Buffer and Eval Env ----------------------------------#
    replay_buffer = ReplayBuffer(state_dim, action_dim, len(dataset["observations"]), args.device)
    state_mean, state_std = replay_buffer.load_d4rl_dataset(dataset)
    # state = replay_buffer._states[20]
    # action = replay_buffer._actions[20]
    eval_env = wrap_env(gym.make(args.env_name), state_mean, state_std)

    # ----------------------------------- Different Seeds Training ----------------------------------#
    for seed in args.seed:

        # ------------------------------- File path and json file ------------------------------- #
        file_name = f"{args.algo_name}/{args.dataset}/{seed}"
        folder_name = os.path.join(args.log_dir, file_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        if os.path.exists(os.path.join(folder_name, 'variant.json')):
            pass
        variant = vars(args)
        variant.update(node=os.uname()[1])
        setup_logger(os.path.basename(folder_name), variant=variant, log_dir=folder_name)

        # --------------------------------------- Set Seed -------------------------------------- #
        set_seed(seed, eval_env)

        # ------------------------------------- Algorithms -------------------------------------- #
        if args.algo_name == 'PLAS':
            policy = train_PLAS(args, eval_env, replay_buffer, state_dim, action_dim, max_action, folder_name)
        if args.algo_name == 'LAPO':
            policy = train_LAPO(args, eval_env, replay_buffer, state_dim, action_dim, max_action, folder_name,
                                min_v, max_v)
        if args.algo_name == 'SPOT':
            policy = train_SPOT(args, eval_env, replay_buffer, state_dim, action_dim, max_action, folder_name)



