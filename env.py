import numpy as np

class CliffWorld():
    def __init__(self, gamma=0.99, episode_cutoff_length=1000, reward_noise=0):
        arr_dict = np.load('cliff_world_env.npz')
        self.P = arr_dict['P']
        self.r = arr_dict['r']
        self.mu = arr_dict['mu']
        self.terminal_states = arr_dict['terminal_states']

        self.state_space = self.r.shape[0]
        self.action_space = self.r.shape[1]

        self.gamma = gamma
        self.episode_cutoff_length = episode_cutoff_length
        self.reward_noise = reward_noise

        self.t = None
        self.state = None

    def reset(self):
        self.t = 0
        self.state = np.random.choice(a=self.state_space, p=self.mu)
        return self.state

    def step(self, action):
        if self.state is None:
            raise Exception('step() used before calling reset()')
        assert action in range(self.action_space)

        reward = self.r[self.state, action] \
            + np.random.normal(loc=0, scale=self.reward_noise)
        self.state = np.random.choice(a=self.state_space,
                                      p=self.P[self.state, action])
        self.t = self.t + 1

        done = 'false'
        if self.state in self.terminal_states:
            done = 'terminal'
        elif self.t > self.episode_cutoff_length:
            done = 'cutoff'

        return self.state, reward, done, {}

    def estimate_advantage(self, pi, num_traj, alg='monte_carlo_avg'):
        qpi = np.zeros(self.P.shape)
        cnt = np.zeros(self.P.shape)

        for traj in range(num_traj):

            # sample a trajectory
            traj = {'state_list': [], 'action_list': [], 'action_prob_list': [],
                    'reward_list': [], 'next_state_list': []}
            state = env.reset()
            done = 'false'
            while done == 'false':
                action = np.random.choice(self.num_actions, p=pi[state])
                action_prob = pi[state, action]
                next_state, reward, done, _ = env.step(action)

                traj['state_list'].append(state)
                traj['action_list'].append(action)
                traj['action_prob_list'].append(action_prob)
                traj['reward_list'].append(reward)
                traj['next_state_list'].append(next_state)

                state = next_state

            G = 0
            for t in range(len(traj['state_list']), -1, -1):
                G = self.gamma * G + traj['reward']
                q[traj['state'], traj['action']] = something_something
                cnt[traj['state'], traj['action']] += 1

        vpi = (qpi * pi).sum(1)
        adv = qpi - vpi

        return adv

#----------------------------------------------------------------------
# Saving the environment data from the original environment :D
#----------------------------------------------------------------------
# from src.envs.gridworld_mdp import cliff_gw
# env = cliff_gw(gamma=0.99)
# np.savez('cliff_world_env', P=env.P, r=env.R, mu=env.p0,
#          terminal_states=env.terminal_states)
