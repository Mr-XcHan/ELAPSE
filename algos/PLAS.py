# TODO 现在用的那个batchnorm出现nan问题，明天检查一下.

import time
import os
from tqdm import tqdm
from logger import logger
from logger import create_stats_ordered_dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import copy

ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")


class Actor(nn.Module):
    def __init__(self, state_dim, latent_dim, max_latent_action):
        super(Actor, self).__init__()

        self.hidden_size = (400, 300)

        self.l1 = nn.Linear(state_dim, self.hidden_size[0])
        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], latent_dim)

        self.max_latent_action = max_latent_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_latent_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.hidden_size = (400, 300)

        self.l1 = nn.Linear(state_dim + action_dim, self.hidden_size[0])
        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], 1)

        self.l4 = nn.Linear(state_dim + action_dim, self.hidden_size[0])
        self.l5 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l6 = nn.Linear(self.hidden_size[1], 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class BatchNorm_v1(nn.Module):
    def __init__(self, dim_z, batch_normalization_weight):
        super(BatchNorm_v1, self).__init__()
        self.dim_z = dim_z
        self.tau = torch.tensor(batch_normalization_weight, requires_grad=False)  # tau : float in range (0,1)

        self.bn = nn.BatchNorm1d(dim_z)
        self.bn.bias.requires_grad = True
        self.bn.weight.requires_grad = False
        with torch.no_grad():
            self.bn.weight.fill_(self.tau)

    def forward(self, x):  # x:(batch_size,dim_z)
        x = self.bn(x)
        return x


class BatchNorm_v2(nn.Module):
    def __init__(self, dim_z, batch_normalization_weight, mu=True):
        super(BatchNorm_v2, self).__init__()
        self.dim_z = dim_z

        self.tau = torch.tensor(batch_normalization_weight, requires_grad=False)  # tau : float in range (0,1)
        self.theta = torch.tensor(0., requires_grad=True)

        self.gamma1 = torch.sqrt(1 - self.tau * torch.sigmoid(self.theta))  # for mu
        self.gamma2 = torch.sqrt(self.tau * torch.sigmoid(self.theta))  # for var

        self.bn = nn.BatchNorm1d(dim_z)
        self.bn.bias.requires_grad = False
        self.bn.weight.requires_grad = True

        if mu:
            with torch.no_grad():
                self.bn.weight.fill_(self.gamma1)
        else:
            with torch.no_grad():
                self.bn.weight.fill_(self.gamma2)

    def forward(self, x):  # x:(batch_size,dim_z)
        x = self.bn(x)
        return x


# Vanilla Variational Auto-Encoder
class BNVAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, batch_normalization_weight, device, hidden_size=750):
        super(BNVAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

        # BatchNorm v1.
        self.bn_mean = BatchNorm_v1(latent_dim, batch_normalization_weight)

        # BatchNorm v2.
        # self.bn_mean = BatchNorm_v2(latent_dim, batch_normalization_weight, mu=True)
        # self.bn_std = BatchNorm_v2(latent_dim, batch_normalization_weight, mu=False)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)
        self.d3 = nn.Linear(hidden_size, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        mean = self.bn_mean(mean)
        # Clamped for numerical stability
        # log_std = self.log_std(z).clamp(-4, 15)
        log_std = self.log_std(z).clamp(-4, 4)
        std = torch.exp(log_std)
        # std = self.bn_std(log_std)

        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state: object, z: object = None, clip: object = None, raw: object = False) -> object:
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device)
            if clip is not None:
                z = z.clamp(-clip, clip)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        a = self.d3(a)
        if raw: return a
        return self.max_action * torch.tanh(a)


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device, hidden_size=750):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)
        self.d3 = nn.Linear(hidden_size, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 4)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state: object, z: object = None, clip: object = None, raw: object = False) -> object:
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device)
            if clip is not None:
                z = z.clamp(-clip, clip)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        a = self.d3(a)
        if raw: return a
        return self.max_action * torch.tanh(a)


class VAEModule(object):
    def __init__(self, args, state_dim, action_dim, latent_dim, max_action):

        self.device = args.device

        if args.batch_normalization:
            self.vae = BNVAE(state_dim, action_dim, latent_dim, max_action, args.batch_normalization_weight,
                             args.device).to(self.device)
        else:
            self.vae = VAE(state_dim, action_dim, latent_dim, max_action, args.device).to(self.device)

        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=args.vae_lr)

    def train(self, args, replay_buffer, folder_name):
        """

        :param args:
        :param replay_buffer:
        :param folder_name:
        :return:
        """
        batch_size = args.batch_size
        iterations = args.vae_iteration
        logs = {'vae_loss': [], 'recon_loss': [], 'kl_loss': [], 'kl_weight': []}
        for i in range(int(iterations / args.save_freq)):
            with tqdm(total=int(args.save_freq), desc='Iteration %d' % ((i+1) * args.save_freq)) \
                    as pbar:
                for i_ite in range(args.save_freq):
                    if args.kl_annealing:
                        if (i * args.save_freq + i_ite) / (iterations * 0.5) < 1:
                            kl_weight = args.vae_kl_weight * (i * args.save_freq + i_ite) / (iterations * 0.5)
                        else:
                            kl_weight = args.vae_kl_weight
                    else:
                        kl_weight = args.vae_kl_weight
                    logs['kl_weight'].append(kl_weight)
                    vae_loss, recon_loss, KL_loss = self.train_step(replay_buffer, batch_size, kl_weight)
                    # print("Iteration:", i * args.save_freq + i_ite, "vae_loss:", vae_loss)
                    logs['vae_loss'].append(vae_loss)
                    logs['recon_loss'].append(recon_loss)
                    logs['kl_loss'].append(KL_loss)

                    if (i_ite + 1) % 1000 == 0:
                        pbar.set_postfix({
                            'Iteration': '%d' % (args.save_freq * i + i_ite + 1),
                            'Training Loss': '%.3f' % vae_loss,
                            'Recon Loss': '%.3f' % recon_loss,
                            'KL loss': '%.3f' % KL_loss
                        })
                        pbar.update(1000)

                self.save('vae_' + str(args.save_freq * (i + 1)), folder_name)
                pickle.dump(logs, open(folder_name + "/vae_logs.p", "wb"))

    def train_step(self, replay_buffer, batch_size, kl_weight):
        state, action, _, _, _ = replay_buffer.sample(batch_size)
        recon, mean, std = self.vae(state, action)
        recon_loss = (torch.sum(F.mse_loss(recon, action, reduction='none'), dim=1).view(-1, 1)).mean()
        KL_loss = (-0.5 * torch.sum((1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)), dim=1).view(-1, 1)).mean()
        vae_loss = recon_loss + KL_loss * kl_weight
        assert not torch.any(torch.isnan(vae_loss)), print(vae_loss)
        self.vae_optimizer.zero_grad()

        vae_loss.backward()
        # for name, parms in self.vae.bn_std.named_parameters():
        #     # for grad in parms.grad.data:
        #     #     if torch.abs(grad) < 1e-4:
        #     #         parms.grad.data = torch.zeros(parms.grad, device=self.device, requires_grad=True)
        #     #         break
        #
        #     assert torch.isnan(parms).sum() == 0, print('-->name:', name, '-->weight:', parms.data, ' -->grad_value:', parms.grad)

        self.vae_optimizer.step()
        return vae_loss.item(), recon_loss.item(), KL_loss.item()

    def save(self, filename, directory):
        torch.save(self.vae.state_dict(), '%s/%s.pth' % (directory, filename))

    def load(self, filename, directory):
        self.vae.load_state_dict(torch.load('%s/%s.pth' % (directory, filename), map_location=self.device))


class Latent(object):
    def __init__(self, vae, state_dim, action_dim, latent_dim, max_latent_action, args):

        self.vae = vae
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.max_latent_action = max_latent_action
        self.gamma = args.gamma
        self.tau = args.tau
        self.lmbda = args.lmbda
        self.policy_noise = args.policy_noise
        self.device = args.device

        self.actor = Actor(state_dim, latent_dim, max_latent_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.vae.decode(state, z=self.actor(state))
        return action.cpu().data.numpy().flatten()

    def train(self, args, eval_env, replay_buffer, folder_name):
        """
        :param args:
        :param eval_env
        :param replay_buffer:
        :param folder_name
        :return:
        """
        self.vae.eval()
        logs = {'normalized_score': []}
        normalized_score = 0

        for i in range(int(args.AC_iteration / args.save_freq)):
            with tqdm(total=int(args.save_freq), desc='Iteration %d' % ((i + 1) * args.save_freq)) \
                    as pbar:
                for i_ite in range(args.save_freq):
                    state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)

                    # Critic Training
                    with torch.no_grad():
                        next_latent_action = self.actor_target(next_state)
                        next_action = self.vae.decode(next_state, z=next_latent_action)
                        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                        target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1 - self.lmbda) \
                                   * torch.max(target_Q1, target_Q2)
                        target_Q = reward + (1 - done) * self.gamma * target_Q

                    current_Q1, current_Q2 = self.critic(state, action)
                    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    # Actor Training
                    latent_actions = self.actor(state)
                    actions = self.vae.decode(state, z=latent_actions)
                    actor_loss = -self.critic.q1(state, actions).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # Update Target Networks
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    if (i_ite + 1) % 1000 == 0:
                        pbar.set_postfix({
                            'Iteration': '%d' % (args.save_freq * i + i_ite + 1),
                            'Critic Loss': '%.3f' % critic_loss,
                            'Actor Loss': '%.3f' % actor_loss,
                            'normalized_score': '%.3f' % normalized_score
                        })
                        pbar.update(1000)

                    if (i_ite + 1) % args.eval_freq == 0:
                        normalized_score = self.eval(eval_env)
                        logs['normalized_score'].append(normalized_score)

            # Save Model
            self.save(str((i + 1) * args.save_freq), folder_name)
            pickle.dump(logs, open(folder_name + "/PLAS_logs.p", "wb"))

    def eval(self, eval_env):
        """

        :param eval_env:
        :return:
        """
        self.actor.eval()
        aver_return = 0
        for i in range(10):
            state, done = eval_env.reset(), False
            while not done:
                state = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
                # latent_action = torch.tensor(np.random.uniform(-2, 2, size=(1, self.latent_dim)), dtype=torch.float32,
                #                              device=self.device)
                latent_action = self.actor(state)
                action = self.vae.decode(state, z=latent_action).cpu().data.numpy().flatten()
                next_state, reward, done, env_infos = eval_env.step(action)
                aver_return += reward
                state = next_state
        self.actor.train()
        return eval_env.get_normalized_score(aver_return / 10) * 100

    def save(self, filename, directory):
        torch.save(self.critic.state_dict(), '%s/critic_%s.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/critic_optimizer_%s.pth' % (directory, filename))

        torch.save(self.actor.state_dict(), '%s/actor_%s.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/actor_optimizer_%s.pth' % (directory, filename))

    def load(self, filename, directory):
        self.critic.load_state_dict(torch.load('%s/critic_%s.pth' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/critic_optimizer_%s.pth' % (directory, filename)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('%s/actor_%s.pth' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/actor_optimizer_%s.pth' % (directory, filename)))
        self.actor_target = copy.deepcopy(self.actor)


def train_PLAS(args, eval_env, replay_buffer, state_dim, action_dim, max_action, folder_name):
    """
    :param args:
    :param eval_env:
    :param replay_buffer:
    :param state_dim:
    :param action_dim:
    :param max_action:
    :param folder_name:
    :return:
    """
    latent_dim = action_dim * args.latent_dim_coff
    vae_trainer = VAEModule(args, state_dim, action_dim, latent_dim, max_action)

    # ---------------------------------- Train VAE or Load VAE model. ------------------------------------ #
    if args.vae_mode == 'train':
        print('*=================================================================*')
        print('*================== PLAS: Start Training VAE =====================*')
        print('*=================================================================*')
        vae_trainer.train(args, replay_buffer, folder_name)
    if args.vae_mode == 'load':
        vae_filename = 'vae_' + str(args.vae_iteration)
        vae_trainer.load(vae_filename, folder_name)
        print('*=================================================================*')
        print('*================= PLAS: Loaded VAE Successfully =================*')
        print('*=================================================================*')

    # ---------------------------------- Train Actor and Critic. ------------------------------------ #
    policy = Latent(vae_trainer.vae, state_dim, action_dim, latent_dim, args.max_latent_action, args)

    # Train Actor Critic Networks.
    print('*=================================================================*')
    print('*============== PLAS: Start Training Actor Critic ================*')
    print('*=================================================================*')
    policy.train(args, eval_env, replay_buffer, folder_name)

    return policy
