import jax

from .algorithms import mdpo, ppo, softmax_ppo, pg
from .algorithms import approx_pi_improve, pi_improve
from src.envs.chain_mdp import get_shamdp
from src.envs.gridworld_mdp import four_rooms_gw


def get_env(env_id, **kwargs):
    env = None
    if env_id == "four_rooms":
        env = four_rooms_gw(**kwargs) # gridworld (not in the paper)
    elif env_id == "shamdp":
        env = get_shamdp(**kwargs) # 
    assert env is not None
    return env


def get_agent(agent_id, approximate_pi):
    pi_fn = None
    if approximate_pi:
        pi_improve_fn = approx_pi_improve
        if agent_id == "ppo":
            pi_fn = ppo
        elif agent_id == "pg":
            pi_fn = pg
    else:
        pi_improve_fn = pi_improve
        if agent_id == "softmax_ppo":
            pi_fn = softmax_ppo
        elif agent_id == "mdpo":
            pi_fn = mdpo
    assert pi_fn is not None

    return jax.partial(pi_improve_fn, pi_fn=pi_fn) # Need to ask what exactly does this function do? [compiles the function]
