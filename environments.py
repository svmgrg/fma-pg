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

    # for computing v^pi = (I - gamma P_pi)^{-1} r_pi
    # P_pi(s' | s) = sum_a P(s' | s, a) * pi(a | s)
    # environment P: (i, j, k)th element = Pr{S' = s_k | S = s_i, A = a_j}
    # policy pi: (i, j)th element = Pr{A = a_j | S = s_i}
    def calc_vpi(self, pi, FLAG_V_S0=False):
        p_pi = np.einsum('xay,xa->xy', self.P, pi)
        r_pi = np.einsum('xa,xa->x', env.r, pi)
        v_pi = np.linalg.solve(np.eye(self.state_space) - self.gamma * p_pi,
                               r_pi)

        if FLAG_V_S0: # calculate v_s0
            v_s0 = np.dot(env.p0, v_pi)
            return v_s0
        else:
            return v_pi

    # for computing q^pi
    # using q^pi(s, a) = r(s, a) + gamma * sum_s' p(s' | s, a) * v^pi(s')
    def calc_qpi(self, pi):
        v_pi = self.calc_vpi(pi)
        q_pi = self.r + self.gamma * np.einsum('xay,y->xa', self.P, v_pi)
        return q_pi

    # computing the normalized occupancy measure d^pi
    # d^pi = (1 - gamma) * mu (I - gamma P_pi)^{-1},
    # where mu is the start state distribution
    def calc_dpi(self, pi):
        p_pi = np.einsum('xay,xa->xy', self.P, pi)
        d_pi = (1 - self.gamma) * \
            np.linalg.solve(np.eye(self.state_space) - self.gamma * p_pi.T,
                            self.mu)
        d_pi /= d_pi.sum() # for addressing numerical errors; really needed?
        return d_pi

    # compute q*
    def calc_q_star(self, num_iters=1000):
        q = np.zeros((self.state_space, self.action_space))
        for i in range(num_iters):
            q_new = self.r + np.einsum("xay,y->xa",
                                       self.gamma * self.P, q.max(1))
            q = q_new.copy()
        return q

    # compute v*
    def calc_v_star(self, num_iters=1000):
        v = np.zeros(self.state_space)
        for i in range(num_iters):
            v_new = self.r + np.einsum("xay,y->xa", self.gamma * self.P, v)
            v_new = v_new.max(1)
            v = v_new.copy()
        return v

    def calc_pi_star(self, num_iters=1000): # just go greedy wrt q_star
        q_star = self.calc_q_star(num_iters=num_iters)
        pi_star = np.zeros((self.state_space, self.action_space))
        pi_star[range(self.state_space), q_star.argmax(1)] = 1
        return pi_star

    def estimate_advantage(self, pi, num_traj, alg='monte_carlo_avg'):
        qpi = np.zeros((self.state_space, self.action_space))
        cnt = np.zeros((self.state_space, self.action_space))

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
