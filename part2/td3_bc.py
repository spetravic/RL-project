import argparse
from torch.nn.modules.loss import MSELoss
import tqdm
import copy
import time
import random

import pybullet_envs
import gym
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########### Helper Functions ##########


def eval_policy(policy, eval_env, mean, std, eval_episodes=10):
    # Runs policy for X episodes and returns average reward
    # A fixed seed is used for the eval environment

    avg_reward = 0.

    for i in range(eval_episodes):
        state, done = eval_env.reset(), False

        while not done:
            state = (state - mean) / std
            action = policy.select_action(state)
            next_state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

            state = next_state

    avg_reward /= eval_episodes

    return {'returns': avg_reward}


########## Define Replay Buffer ##########
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device),
        )


    # Added for TD3_BC (eq. 4 in the paper)
    def normalization(self, epsilon=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + epsilon

        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std

        return mean, std

    # Added for TD3_BC
    def use_offline_data(self, data):
        self.state = data['states']
        self.action = data['actions']
        self.next_state = data['next_states']
        self.reward = data['rewards']
        self.not_done = data['not_dones']

        self.size = self.state.shape[0]


########## Define Agent ##########
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


########## TD3 ##########
class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,

        # Added for TD3_BC
        alpha = 2.5 # value used in the original paper

    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        # Added for TD3_BC
        self.alpha = alpha

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = batch

        # Update the critic network
        # Hint: You can use clamp() to clip values
        # Hint: Like before, pay attention to which variable should be detached.
        with torch.no_grad():
            e = torch.zeros_like(action).normal_(0, self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) +e).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target.forward(next_state, next_action)
            target_Q = reward + self.discount * torch.min(target_Q1, target_Q2) * not_done

        current_Q1, current_Q2 = self.critic(state, action)

        MSE = torch.nn.MSELoss()
        critic_loss = MSE(current_Q1, target_Q) + MSE(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            Q = self.critic.forward(state, self.actor(state))[0]
            l = self.alpha / torch.abs(Q).mean().detach() # eq. 5 in the paper

            actor_loss = - (l * Q.mean() - MSE(self.actor(state), action)) # eq. 3 in the paper

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models, both actor and critic
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"critic_loss": critic_loss.item(),
                "critic": current_Q1.mean().item()}

    def save(self, filename):
        torch.save({'critic': self.critic.state_dict(),
                    'actor': self.actor.state_dict(), }, filename + "_td3_bc.pth")

    def load(self, filename):
        policy_dicts = torch.load(filename + "_td3.pth")

        self.critic.load_state_dict(policy_dicts['critic'])
        self.target_critic = copy.deepcopy(self.critic)

        self.actor.load_state_dict(policy_dicts['actor'])
        self.target_actor = copy.deepcopy(self.actor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo_name', default='TD3_BC')
    parser.add_argument('--env', default='HalfCheetahBulletEnv-v0')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--max_timesteps', type=int, default=500000,
                        help='total timesteps of the experiments')
    parser.add_argument('--n_random_timesteps', type=int, default=10000,
                        help='num of inital random data to pre-fill the replay buffer')
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--eval_freq", default=5000, type=int)
    # Algorithm specific arguments
    parser.add_argument("--expl_noise", default=0.1, type=float,
                        help="Std of Gaussian exploration noise")
    parser.add_argument("--discount", default=0.99, type=float,
                        help="Discount factor.")
    parser.add_argument("--tau", default=0.005, type=float,
                        help="Target network update rate")
    parser.add_argument("--policy_noise", default=0.2, type=float,
                        help="Noise added to target policy during critic update")
    parser.add_argument("--noise_clip", default=0.5, type=float,
                        help="Range to clip target policy noise")
    parser.add_argument("--policy_freq", default=2, type=int,
                        help="Frequency of delayed policy updates")
    # options
    parser.add_argument('--seed', type=int, default=0,
                        help='seed of the experiment')
    parser.add_argument("--save_model", action="store_true")

    # Added for TD3_BC
    parser.add_argument("--alpha", default=2.5, type=float,
                        help="Hyperparameter for the normalization term, alpha=2.5 in the original paper")
    parser.add_argument("--ratio", default=1, type=float,
                        help="Percentage of data")
    parser.add_argument("--dataset", default="halfcheetah_mixed.pickle")
    
    
    args = parser.parse_args()
    if args.seed == 0:
        args.seed = int(time.time())

    experiment_name = str(args.env) + '_' + str(args.algo_name) + \
        '_' + str(args.seed) + '_' + str(int(time.time()))

    wandb.init(project="rl_project", config=vars(args), name=experiment_name)

    # Init env and seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    eval_env = gym.make(args.env)
    eval_env.seed(args.seed + 100)
    eval_env.action_space.seed(args.seed + 100)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    # Added for TD3_BC
    # Load dataset and give to function
    with open(args.dataset, 'rb') as f:
        data = pickle.load(f)

    n = int(args.ratio * len(data['states']))

    data_ratio = {}
    for k in data.keys():
        data_ratio[k] = data[k][:n]

    replay_buffer.use_offline_data(data_ratio)
    
    mean, std = replay_buffer.normalization()

    # init td3
    td3_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise,
        "noise_clip": args.noise_clip,
        "policy_freq": args.policy_freq,

        # Added for TD3_BC
        "alpha": args.alpha
    }

    td3 = TD3(**td3_kwargs)

    # Changed for TD3_BC
    for t in tqdm.tqdm(range(args.max_timesteps)):
        policy_update_info = td3.train(replay_buffer.sample(args.batch_size))
        wandb.log({"train/": policy_update_info})

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            eval_info = eval_policy(td3, eval_env, mean, std)
            eval_info.update({'timesteps': t})
            print("Time steps:", t, "Eval_info:", eval_info)
            wandb.log({"eval/": eval_info})

    if args.save_model:
        td3.save("./" + str(experiment_name) + "_" + "ratio-" + str(args.ratio))

    env.close()
