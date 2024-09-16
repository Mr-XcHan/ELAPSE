import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
import torch.distributions as td
import pickle


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
    ):
        super(Actor, self).__init__()

        head = nn.Linear(256, action_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            head,
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.hidden_size = (256, 256)

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
        if z.any() == 0:
            print('z all is 0')

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

    def importance_sampling_estimator(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        vae_kl_weight,
        num_samples: int = 500,
    ) -> torch.Tensor:
        # * num_samples correspond to num of samples L in the paper
        # * note that for exact value for \hat \log \pi_\beta in the paper
        # we also need **an expection over L samples**
        mean, std = self.encode(state, action)

        mean_enc = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_enc = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_enc + std_enc * torch.randn_like(std_enc)  # [B x S x D]

        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        mean_dec = self.decode(state, z)
        std_dec = np.sqrt(vae_kl_weight / 4)

        # Find q(z|x)
        log_qzx = td.Normal(loc=mean_enc, scale=std_enc).log_prob(z)
        # Find p(z)
        mu_prior = torch.zeros_like(z).to(self.device)
        std_prior = torch.ones_like(z).to(self.device)
        log_pz = td.Normal(loc=mu_prior, scale=std_prior).log_prob(z)
        # Find p(x|z)
        std_dec = torch.ones_like(mean_dec).to(self.device) * std_dec
        log_pxz = td.Normal(loc=mean_dec, scale=std_dec).log_prob(action)

        w = log_pxz.sum(-1) + log_pz.sum(-1) - log_qzx.sum(-1)
        ll = w.logsumexp(dim=-1) - np.log(num_samples)
        return ll

    def encode(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        mean = self.bn_mean(mean)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 4)
        std = torch.exp(log_std)
        return mean, std

    def decode(
        self,
        state: torch.Tensor,
        z: torch.Tensor = None,
    ) -> torch.Tensor:
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = (
                torch.randn((state.shape[0], self.latent_dim))
                .to(self.device)
                .clamp(-0.5, 0.5)
            )
        a = F.relu(self.d1(torch.cat([state, z], -1)))
        a = F.relu(self.d2(a))
        a = self.d3(a)
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

    def importance_sampling_estimator(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        vae_kl_weight,
        num_samples: int = 500,
    ) -> torch.Tensor:
        # * num_samples correspond to num of samples L in the paper
        # * note that for exact value for \hat \log \pi_\beta in the paper
        # we also need **an expection over L samples**
        mean, std = self.encode(state, action)

        mean_enc = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_enc = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_enc + std_enc * torch.randn_like(std_enc)  # [B x S x D]

        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        mean_dec = self.decode(state, z)
        std_dec = np.sqrt(vae_kl_weight / 4)

        # Find q(z|x)
        log_qzx = td.Normal(loc=mean_enc, scale=std_enc).log_prob(z)
        # Find p(z)
        mu_prior = torch.zeros_like(z).to(self.device)
        std_prior = torch.ones_like(z).to(self.device)
        log_pz = td.Normal(loc=mu_prior, scale=std_prior).log_prob(z)
        # Find p(x|z)
        std_dec = torch.ones_like(mean_dec).to(self.device) * std_dec
        log_pxz = td.Normal(loc=mean_dec, scale=std_dec).log_prob(action)

        w = log_pxz.sum(-1) + log_pz.sum(-1) - log_qzx.sum(-1)
        ll = w.logsumexp(dim=-1) - np.log(num_samples)
        return ll

    def encode(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 4)
        std = torch.exp(log_std)
        return mean, std


    def decode(
        self,
        state: torch.Tensor,
        z: torch.Tensor = None,
    ) -> torch.Tensor:
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = (
                torch.randn((state.shape[0], self.latent_dim))
                .to(self.device)
                .clamp(-0.5, 0.5)
            )

        a = F.relu(self.d1(torch.cat([state, z], -1)))
        a = F.relu(self.d2(a))
        a = self.d3(a)
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
        :param dataset:
        :param folder_name:
        :return:
        """
        batch_size = args.batch_size
        iterations = args.vae_iteration
        logs = {'vae_loss': [], 'recon_loss': [], 'kl_loss': []}

        for i in range(int(iterations/args.save_freq)):
            with tqdm(total=int(args.save_freq), desc='Iteration %d' % ((i + 1) * args.save_freq))\
                    as pbar:
                for i_ite in range(args.save_freq):
                    if args.kl_annealing:
                        if (i * args.save_freq + i_ite) / (iterations * 0.5) < 1:
                            kl_weight = args.vae_kl_weight * (i * args.save_freq + i_ite) / (iterations * 0.5)
                        else:
                            kl_weight = args.vae_kl_weight
                    else:
                        kl_weight = args.vae_kl_weight
                    vae_loss, recon_loss, KL_loss = self.train_step(replay_buffer, batch_size, kl_weight)
                    logs['vae_loss'].append(vae_loss)
                    logs['recon_loss'].append(recon_loss)
                    logs['kl_loss'].append(KL_loss)

                    if (i_ite + 1) % 1000 == 0:
                        pbar.set_postfix({
                            'Iteration': '%d' % (args.save_freq * i + i_ite + 1),
                            'VAE Loss': '%.3f' % vae_loss,
                            'Recon Loss': '%.3f' % recon_loss,
                            'KL loss': '%.3f' % KL_loss
                        })
                        pbar.update(1000)

                self.save('vae_' + str(args.save_freq * (i+1)), folder_name)
                pickle.dump(logs, open(folder_name + "/vae_logs.p", "wb"))

    def train_step(self, replay_buffer, batch_size, kl_weight):
        state, action, _, _, _ = replay_buffer.sample(batch_size)
        recon, mean, std = self.vae(state, action)
        recon_loss = (torch.sum(F.mse_loss(recon, action, reduction='none'), dim=1).view(-1, 1)).mean()
        KL_loss = (-0.5 * torch.sum((1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)), dim=1).view(-1, 1)).mean()
        vae_loss = recon_loss + kl_weight * KL_loss
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        return vae_loss.item(), recon_loss.item(), KL_loss.item()

    def save(self, filename, directory):
        torch.save(self.vae.state_dict(), '%s/%s.pth' % (directory, filename))

    def load(self, filename, directory):
        self.vae.load_state_dict(torch.load('%s/%s.pth' % (directory, filename), map_location=self.device))


class SPOT:
    def __init__(self, vae, state_dim, action_dim, latent_dim, max_action, args):

        self.vae = vae
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.max_action = max_action
        self.max_latent_action = args.max_latent_action

        self.gamma = args.gamma
        self.tau = args.tau
        self.vae_kl_weight = args.vae_kl_weight
        self.SPOT_lmbda = args.SPOT_lmbda
        self.num_samples = args.SPOT_num_samples
        self.policy_noise = args.policy_noise
        self.policy_freq = args.policy_freq
        self.iwae = args.SPOT_iwae

        self.device = args.device

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

    def elbo_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        vae_kl_weight: float,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Note: elbo_loss one is proportional to elbo_estimator
        i.e. there exist a>0 and b, elbo_loss = a * (-elbo_estimator) + b
        """
        mean, std = self.vae.encode(state, action)

        mean_s = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_s = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_s + std_s * torch.randn_like(std_s)

        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        u = self.vae.decode(state, z)
        recon_loss = ((u - action) ** 2).mean(dim=(1, 2))

        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        vae_loss = recon_loss + vae_kl_weight * KL_loss
        return vae_loss

    def iwae_loss(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            vae_kl_weight,
            num_samples: int = 10,
    ) -> torch.Tensor:
        ll = self.vae.importance_sampling_estimator(state, action, vae_kl_weight, num_samples)
        return -ll

    def train(self, args, env, replay_buffer, folder_name):

        self.vae.eval()
        logs = {'critic_loss': [], 'actor_loss': [], 'regularization': [], 'normalized_score': []}
        normalized_score = 0

        for i in range(int(args.AC_iteration / args.save_freq)):
            with tqdm(total=int(args.save_freq), desc='Iteration %d' % ((i + 1) * args.save_freq)) \
                    as pbar:
                for i_ite in range(args.save_freq):
                    state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)
                    # Critic Training
                    with torch.no_grad():
                        next_action = self.actor_target(next_state)
                        # noise = (torch.randn_like(next_action) * self.policy_noise).clamp(
                        #     -args.policy_noise * self.max_action, args.policy_noise * self.max_action)
                        # next_action += noise

                        # Compute the target Q value
                        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                        target_Q = torch.min(target_Q1, target_Q2)
                        target_Q = reward + (1 - done) * self.gamma * target_Q

                    # Get current Q estimates
                    current_Q1, current_Q2 = self.critic(state, action)

                    # Compute critic loss
                    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()
                    logs["critic_loss"].append(critic_loss.item())

                    # Delayed actor updates
                    if i_ite % self.policy_freq == 0:
                        # Compute actor loss
                        pi = self.actor(state)
                        q = self.critic.q1(state, pi)

                        if self.iwae:
                            neg_log_beta = self.iwae_loss(state, pi, self.vae_kl_weight, self.num_samples).mean()
                        else:
                            neg_log_beta = self.elbo_loss(state, pi, self.vae_kl_weight, self.num_samples).mean()

                        norm_q = 1 / q.abs().mean().detach()

                        actor_loss = -norm_q * q.mean() + self.SPOT_lmbda * neg_log_beta

                        # Optimize the actor
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()
                        logs["actor_loss"].append(actor_loss.item())
                        logs["regularization"].append(neg_log_beta.item())

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
                            'Normalized_score': '%.3f' % normalized_score
                        })
                        pbar.update(1000)

                    if (i_ite + 1) % args.eval_freq == 0:
                        normalized_score = self.eval(env)
                        logs['normalized_score'].append(normalized_score)

            # Save Model
            self.save(str((i + 1)*args.save_freq), folder_name)
            pickle.dump(logs, open(folder_name + "/SPOT_logs.p", "wb"))

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state).cpu().data.numpy().flatten()

        return action

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
                action = self.actor(state).cpu().data.numpy().flatten()
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

    def load(self, filename, directory):
        self.critic.load_state_dict(torch.load('%s/critic_%s.pth' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/critic_optimizer_%s.pth' % (directory, filename)))
        self.critic_target.load_state_dict(torch.load('%s/critic_target_%s.pth' % (directory, filename)))

        self.actor.load_state_dict(torch.load('%s/actor_%s.pth' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/actor_optimizer_%s.pth' % (directory, filename)))
        self.actor_target.load_state_dict(torch.load('%s/actor_target_%s.pth' % (directory, filename)))


def train_SPOT(args, eval_env, replay_buffer, state_dim, action_dim, max_action, folder_name):
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
        print('*================== SPOT: Start Training VAE =====================*')
        print('*=================================================================*')
        vae_trainer.train(args, replay_buffer, folder_name)
    if args.vae_mode == 'load':
        vae_filename = 'vae_' + str(args.vae_iteration)
        vae_trainer.load(vae_filename, folder_name)
        print('*=================================================================*')
        print('*================= SPOT: Loaded VAE Successfully =================*')
        print('*=================================================================*')

    # ---------------------------------- Train Actor and Critic. ------------------------------------ #
    policy = SPOT(vae_trainer.vae, state_dim, action_dim, latent_dim, max_action, args)

    print('*=================================================================*')
    print('*============== SPOT: Start Training Actor Critic ================*')
    print('*=================================================================*')
    policy.train(args, eval_env, replay_buffer, folder_name)
