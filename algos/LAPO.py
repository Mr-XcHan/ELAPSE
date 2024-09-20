import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
import pickle


class Actor(nn.Module):
    def __init__(self, state_dim, latent_dim, max_action):
        super(Actor, self).__init__()
        hidden_size = (256, 256, 256)

        self.pi1 = nn.Linear(state_dim, hidden_size[0])
        self.pi2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.pi3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.pi4 = nn.Linear(hidden_size[2], latent_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.pi1(state))
        a = F.relu(self.pi2(a))
        a = F.relu(self.pi3(a))
        a = self.pi4(a)
        a = self.max_action * torch.tanh(a)

        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        hidden_size = (256, 256, 256)

        self.l1 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.l4 = nn.Linear(hidden_size[2], 1)

        self.l5 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.l6 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l7 = nn.Linear(hidden_size[1], hidden_size[2])
        self.l8 = nn.Linear(hidden_size[2], 1)

        self.v1 = nn.Linear(state_dim, hidden_size[0])
        self.v2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.v3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.v4 = nn.Linear(hidden_size[2], 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = (self.l4(q1))

        q2 = F.relu(self.l5(torch.cat([state, action], 1)))
        q2 = F.relu(self.l6(q2))
        q2 = F.relu(self.l7(q2))
        q2 = (self.l8(q2))
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = (self.l4(q1))
        return q1

    def v(self, state):
        v = F.relu(self.v1(state))
        v = F.relu(self.v2(v))
        v = F.relu(self.v3(v))
        v = (self.v4(v))
        return v


class BatchNorm_v1(nn.Module):
    def __init__(self, dim_z, batch_normalization_weight):
        super(BatchNorm_v1, self).__init__()
        self.dim_z = dim_z
        self.tau = torch.tensor(batch_normalization_weight, requires_grad=False)  # tau : float in range (0,1)

        self.bn = nn.BatchNorm1d(dim_z)
        self.bn.bias.requires_grad = False
        self.bn.weight.requires_grad = True
        with torch.no_grad():
            # self.bn.weight.fill_(self.tau)
            self.bn.bias.fill_(self.tau)
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


class BNVAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, batch_normalization_weight, device):
        super(BNVAE, self).__init__()
        hidden_size = (256, 256, 256)

        self.e1 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.e2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.e3 = nn.Linear(hidden_size[1], hidden_size[2])

        self.mean = nn.Linear(hidden_size[2], latent_dim)
        self.log_std = nn.Linear(hidden_size[2], latent_dim)

        # BatchNorm v1.
        self.bn_mean = BatchNorm_v1(latent_dim, batch_normalization_weight)

        # BatchNorm v2.
        # self.bn_mean = BatchNorm_v2(latent_dim, batch_normalization_weight, mu=True)
        # self.bn_std = BatchNorm_v2(latent_dim, batch_normalization_weight, mu=False)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_size[0])
        self.d2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.d3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.d4 = nn.Linear(hidden_size[2], action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))
        z = F.relu(self.e3(z))

        mean = self.mean(z)
        mean = self.bn_mean(mean)
        # Clamped for numerical stability
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
        a = F.relu(self.d3(a))
        a = self.d4(a)
        if raw: return a
        return self.max_action * torch.tanh(a)


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        hidden_size = (256, 256, 256)

        self.e1 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.e2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.e3 = nn.Linear(hidden_size[1], hidden_size[2])

        self.mean = nn.Linear(hidden_size[2], latent_dim)
        self.log_std = nn.Linear(hidden_size[2], latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_size[0])
        self.d2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.d3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.d4 = nn.Linear(hidden_size[2], action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))
        z = F.relu(self.e3(z))

        mean = self.mean(z)

        # Clamped for numerical stability
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
        a = F.relu(self.d3(a))
        a = self.d4(a)
        if raw: return a
        return self.max_action * torch.tanh(a)


class LAPO(object):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, min_v, max_v, replay_buffer, args):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.max_action = max_action
        self.max_latent_action = args.max_latent_action

        self.gamma = args.gamma
        self.tau = args.tau
        self.lmbda = args.lmbda
        self.policy_noise = args.policy_noise

        self.expectile = 0.9
        self.kl_weight = args.vae_kl_weight
        self.no_noise = True  # add noise to latent space or not.

        self.replay_buffer = replay_buffer
        self.min_v, self.max_v = min_v, max_v

        self.device = args.device

        if args.batch_normalization:
            self.vae = BNVAE(state_dim, action_dim, latent_dim, max_action, args.batch_normalization_weight,
                             args.device).to(self.device)
        else:
            self.vae = VAE(state_dim, action_dim, latent_dim, max_action, args.device).to(self.device)

        self.vae_target = copy.deepcopy(self.vae)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=args.vae_lr)

        self.actor = Actor(state_dim, latent_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.vae_target.decode(state, z=self.actor(state))
        return action.cpu().data.numpy().flatten()

    def train(self, args, eval_env, folder_name):
        """

        :param args:
        :param eval_env:
        :param folder_name:
        :return:
        """
        normalized_score = 0
        logs = {'vae_loss': [], 'kl_loss': [], 'kl_weight': [], 'critic_loss': [], 'actor_loss': [],
                'normalized_score': []}

        for i in range(int(args.AC_iteration / args.save_freq)):
            with tqdm(total=int(args.save_freq), desc='Iteration %d' % (i * args.save_freq)) \
                    as pbar:
                for i_ite in range(args.save_freq):
                    state, action, reward, next_state, done = self.replay_buffer.sample(args.batch_size)

                    # Critic Training
                    with torch.no_grad():
                        next_target_v = self.critic.v(next_state)
                        target_Q = reward + (1 - done) * self.gamma * next_target_v

                        latent_target_action = self.actor_target(state)
                        # noise = (torch.randn_like(latent_target_action) * self.policy_noise).clamp(
                        #     -args.policy_noise * self.max_action, args.policy_noise * self.max_action)
                        # latent_target_action += noise
                        decode_action = self.vae_target.decode(state, z=latent_target_action)
                        target_Q1, target_Q2 = self.critic_target(state, decode_action)
                        target_v = self.lmbda * torch.min(target_Q1, target_Q2) + (1 - self.lmbda) \
                                   * torch.max(target_Q1, target_Q2)

                    current_Q1, current_Q2 = self.critic(state, action)
                    current_v = self.critic.v(state)

                    v_loss = F.mse_loss(current_v, target_v.clamp(self.min_v, self.max_v))
                    critic_loss_1 = F.mse_loss(current_Q1, target_Q)
                    critic_loss_2 = F.mse_loss(current_Q2, target_Q)
                    critic_loss = critic_loss_1 + critic_loss_2 + v_loss
                    logs['critic_loss'].append(critic_loss.item())
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    # compute adv and weight
                    current_v = self.critic.v(state)
                    current_q = self.lmbda * torch.min(current_Q1, current_Q2) + (1 - self.lmbda) \
                                   * torch.max(current_Q1, current_Q2)
                    adv = (current_q - current_v)

                    weights = torch.where(adv > 0, self.expectile, 1 - self.expectile)

                    # train weighted CVAE - cyclical annealing
                    if args.kl_annealing:
                        if i_ite == 0:
                            kl_weight = 0
                        else:
                            kl_weight = (i_ite / args.save_freq) * args.vae_kl_weight * 1.5
                    else:
                        kl_weight = args.vae_kl_weight

                    recon, mean, std = self.vae(state, action)
                    recon_loss = (torch.sum(F.mse_loss(recon, action, reduction='none'), dim=1).view(-1, 1))
                    KL_loss = -0.5 * torch.sum((1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)),
                                                dim=1).view(-1, 1)
                    vae_loss = ((recon_loss + KL_loss * kl_weight) * weights.detach()).mean()
                    self.vae_optimizer.zero_grad()
                    vae_loss.backward()
                    self.vae_optimizer.step()
                    logs['vae_loss'].append(vae_loss.item())
                    logs['kl_loss'].append(KL_loss.mean().item())
                    logs['kl_weight'].append(self.kl_weight)

                    # train latent policy
                    latent_actor_action = self.actor(state)
                    actor_action = self.vae_target.decode(state, z=latent_actor_action)
                    q_pi = self.critic.q1(state, actor_action)

                    actor_loss = -q_pi.mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    logs['actor_loss'].append(actor_loss.item())

                    # Update Target Networks
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    for param, target_param in zip(self.vae.parameters(), self.vae_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    if (i_ite + 1) % 1000 == 0:
                        pbar.set_postfix({
                            'Iteration': '%d' % (args.save_freq * i + i_ite + 1),
                            'VAE Loss': '%.3f' % vae_loss,
                            'KL Loss': '%.3f' % KL_loss.mean(),
                            'Critic Loss': '%.3f' % critic_loss,
                            'Actor Loss': '%.3f' % actor_loss,
                            'Normalized_score': '%.3f' % normalized_score
                        })
                        pbar.update(1000)

                    if (i_ite + 1) % args.eval_freq == 0:
                        normalized_score = self.eval(eval_env)
                        logs['normalized_score'].append(normalized_score)

                assert (np.abs(np.mean(target_Q.cpu().data.numpy())) < 1e6)

            # Save Model
            self.save(str((i + 1)*args.save_freq), folder_name)
            pickle.dump(logs, open(folder_name + "/LAPO_logs.p", "wb"))

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
                latent_action = self.actor(state)
                action = self.vae_target.decode(state, z=latent_action).cpu().data.numpy().flatten()
                next_state, reward, done, env_infos = eval_env.step(action)
                aver_return += reward
                state = next_state
        self.actor.train()
        return eval_env.get_normalized_score(aver_return / 10) * 100

    def save(self, filename, directory):
        torch.save(self.critic.state_dict(), '%s/critic_%s.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/critic_optimizer_%s.pth' % (directory, filename))
        torch.save(self.critic_target.state_dict(), '%s/critic_target_%s.pth' % (directory, filename))

        torch.save(self.actor.state_dict(), '%s/actor_%s.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/actor_optimizer_%s.pth' % (directory, filename))
        torch.save(self.actor_target.state_dict(), '%s/actor_target_%s.pth' % (directory, filename))

        torch.save(self.vae.state_dict(), '%s/vae_%s.pth' % (directory, filename))
        torch.save(self.vae_optimizer.state_dict(), '%s/vae_optimizer_%s.pth' % (directory, filename))
        torch.save(self.vae_target.state_dict(), '%s/vae_target_%s.pth' % (directory, filename))

    def load(self, filename, directory):
        self.critic.load_state_dict(torch.load('%s/critic_%s.pth' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/critic_optimizer_%s.pth' % (directory, filename)))
        self.critic_target.load_state_dict(torch.load('%s/critic_target_%s.pth' % (directory, filename)))

        self.actor.load_state_dict(torch.load('%s/actor_%s.pth' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/actor_optimizer_%s.pth' % (directory, filename)))
        self.actor_target.load_state_dict(torch.load('%s/actor_target_%s.pth' % (directory, filename)))

        self.vae.load_state_dict(torch.load('%s/vae_%s.pth' % (directory, filename)))
        self.vae_optimizer.load_state_dict(torch.load('%s/vae_optimizer_%s.pth' % (directory, filename)))
        self.vae_target.load_state_dict(torch.load('%s/vae_target_%s.pth' % (directory, filename)))


def train_LAPO(args, eval_env, replay_buffer, state_dim, action_dim, max_action, folder_name, min_v, max_v):
    """

    :param args:
    :param eval_env:
    :param replay_buffer:
    :param state_dim:
    :param action_dim:
    :param max_action:
    :param folder_name:
    :param min_v:
    :param max_v:
    :return:
    """

    latent_dim = action_dim * args.latent_dim_coff

    policy = LAPO(state_dim, action_dim, latent_dim, max_action, min_v, max_v, replay_buffer, args)

    # Train VAE and Actor Critic Networks.
    print('*=================================================================*')
    print('*============== LAPO: Start Training Actor Critic ================*')
    print('*=================================================================*')
    policy.train(args, eval_env, folder_name)

    return policy

