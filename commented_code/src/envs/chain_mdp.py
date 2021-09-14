import numpy as np
from emdp.chainworld import build_chain_MDP
from emdp.common import MDP


def get_n_state_chain(n_states=2):
    state_distribution = np.zeros(shape=(n_states,))
    state_distribution[0] = 1
    reward_spec = np.zeros(shape=(n_states, 2))
    reward_spec[-1, 1] = 1.
    reward_spec[-1, 0] = 1.
    mdp = build_chain_MDP(n_states=n_states, p_success=1.0,
                          reward_spec=reward_spec,
                          starting_distribution=state_distribution,
                          terminal_states=[n_states], gamma=0.9)
    mdp.P[1, 0, 0] = 0
    mdp.P[1, 0, 1] = 1
    mdp.P[1, 1, 1] = 1
    return mdp


def get_shamdp(horizon=20, c=1.6):
    gamma = horizon / (horizon + 1)
    # going right except in the absorbing state incurs a penalty
    right_penalty = -gamma ** (horizon // c)
    P, r = _build_shamdp(horizon, 1, n_actions=4, right_penalty=right_penalty)
    n_states = P.shape[0]

    initial_state_distribution = np.zeros(n_states)
    initial_state_distribution[0] = 1.
    return MDP(P, r, gamma, initial_state_distribution, terminal_states=[n_states - 2])


def _build_shamdp(horizon, end_reward, n_actions=4, right_penalty=0.):
    n_states = horizon + 2
    P = np.zeros((n_states, n_actions, n_states))  # (s, a, s')
    r = np.zeros((n_states, n_actions))

    # taking action 0 at absorbing state gives you reward 1
    r[-1, 0] = end_reward

    # optional penalty of going right at every state but the last
    r[:-1, 0] = right_penalty

    # populate the transition matrix
    # forward actions
    for s in range(n_states - 1):
        P[s, 0, s + 1] = 1.
    P[n_states - 1, :, n_states - 1] = 1.  # irrespective of the action, you end up in the last state forever
    # backward actions
    for s in range(1, n_states - 1):
        P[s, 1:, s - 1] = 1.
    P[0, 1:, 0] = 1.
    return P, r
