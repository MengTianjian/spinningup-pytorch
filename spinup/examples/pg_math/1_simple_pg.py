import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from gym.spaces import Discrete, Box

class MLP(nn.Module):
    def __init__(self, obs_dim, sizes, activation=nn.Tanh, output_activation=None):
        super(MLP, self).__init__()
        for i, size in enumerate(sizes[:-1]):
            self.add_module('dense_%d'%i, nn.Linear(obs_dim, size))
            if activation is not None:
                self.add_module('act_%d'%i, activation())
        if len(sizes) > 1:
            self.add_module('final_dense', nn.Linear(sizes[-2], sizes[-1]))
        else:
            self.add_module('final_dense', nn.Linear(obs_dim, sizes[-1]))
        if output_activation is not None:
            self.add_module('final_act', output_activation())

    def forward(self, x):
        for name, mod in self.named_children():
            x = mod(x)
        return x

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    model = MLP(obs_dim, sizes=hidden_sizes+[n_acts])

    # make train op
    train_optim = torch.optim.Adam(model.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_log_probs = []    # for log probabilities
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            logits = model(torch.tensor(obs).view(1,-1).float())
            prob = F.softmax(logits, dim=-1)
            act = torch.multinomial(prob, 1)[0]
            batch_log_probs.append(F.cross_entropy(logits, act))
            obs, rew, done, _ = env.step(act.item())

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        train_optim.zero_grad()
        batch_loss = []
        for i in range(len(batch_acts)):
            batch_loss.append(batch_log_probs[i]*batch_weights[i])
        loss = sum(batch_loss)/len(batch_loss)
        loss.backward()
        train_optim.step()
        return loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
