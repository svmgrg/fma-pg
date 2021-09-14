# Questions for Simone
# - Comments in the code - which algorithms/environment are supported 
# - How is the logging done. Which plots to check?
# - Run sweep + logging mechanics
# - Reproduce the plot in the paper
# - Where exactly is the overflow in the probs that lead to saturation/zeros and eventually to poor exploration. 
# - What is missing in the FMAPG implementation?? If m -> \infty, with appropriate parametric step-size, does FMAPG recover the exact
# updates in the tabular setting?

import jax.numpy as jnp
import numpy as np

import config
import src.algorithms
from src import get_env, get_agent
from src.utils.plot_utils import gridworld_plot_sa, plot_vf


def get_initial_policy(env):
    pi = np.ones(shape=(env.state_space, env.action_space))
    pi /= pi.sum(1, keepdims=True)
    policy = jnp.array(pi)
    return policy


def stop_criterion(step):
    return step > config.max_steps


# pi_fn = pi_improve (exact, approximate) + method (ppo, mdpo, pg)
def train(env, pi_fn):
    policy = get_initial_policy(env) # initialize policy

    ### How do you pass the FMAPG parameters? [Accessed directly from config]
    pi, v, *_ = src.algorithms.policy_iteration(env, pi_opt=pi_fn, stop_criterion=stop_criterion, eta=config.eta,
                                                policy=policy) # gets the pis, vs  


def main():

    # returns four rooms (grid world), shamdp (which of these environments were used in the paper) Matches gridworld/chain in the envs folder??
    env = get_env(config.env_id, **config.env_kwargs)
    
    # returns compiled functiion. pi_improve (exact, approximate) + method (ppo, mdpo, pg) What is pg_clip in the sweep_params file??
    policy_update_fn = get_agent(config.agent_id, config.approximate_pi)

    train(env, policy_update_fn)
    config.tb.run.finish()

    ### Where is the plotting/storing of the results? Handled by TB? How to avoid writing the run number (what in the code does this?)
    # command line arguments?



if __name__ == '__main__':
    main()
