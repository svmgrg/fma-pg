import os

import jax.numpy as np
import matplotlib.pyplot as plt

import config


#### Need to ask this. What does each function do?


def chain_plot_vf(vf, title=None, ax=None, log_plot=True, step=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    # ax.scatter(np.arange(env.state_space), vf, label=label)
    ax.plot(vf, 'o')
    ax.set_xlabel('state_idx')
    ax.set_ylabel('v(s)')
    # ax.set_title(title, fontdict={'fontsize': 8, 'fontweight': 'medium'})
    tag = f"plots/{title}"
    if log_plot:
        plt.savefig(tag.replace(".", "_"))
        assert step is not None
        config.tb.plot(tag, plt, step)
        plt.clf()
    return fig


def chain_plot_params(env, data, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.scatter(np.arange(env.action_space), data.squeeze())
    ax.set_title(title, fontdict={'fontsize': 8, 'fontweight': 'medium'})
    ax.set_xlabel("actions")
    ax.set_ylabel("p(a)")
    ax.set_title(title, fontdict={'fontsize': 8, 'fontweight': 'medium'})


def chain_plot_sa(env, data, title, ax=None, label=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.plot(data[:, 0], label=label)
    ax.set_xlabel("states")
    ax.set_ylabel("p(right)")
    return fig


def gridworld_plot_sa(env, data, title, ax=None, frame=(0, 0, 0, 0), step=None, log_plot=False):
    """
    This is going to generate a quiver plot to visualize the policy graphically.
    It is useful to see all the probabilities assigned to the four possible actions
    in each state
    """
    if ax is None:
        ax = plt.gca()
    num_cols = env.ncol if hasattr(env, "ncol") else env.size
    num_rows = env.ncol if hasattr(env, "nrow") else env.size

    num_obs, num_actions = data.shape

    direction = [
        np.array((-1, 0)),  # left
        np.array((1, 0)),  # right
        np.array((0, 1)),  # up
        np.array((0, -1)),  # down
    ]

    x, y = np.meshgrid(np.arange(env.size), np.arange(env.size))
    x, y = x.flatten(), y.flatten()

    for base, a in zip(direction, range(num_actions)):
        quivers = np.einsum("d,m->md", base, data[:, a])

        pos = data[:, a] > 0
        ax.quiver(x[pos], y[pos], *quivers[pos].T, units='xy', scale=2.0, color='g')

        pos = data[:, a] < 0
        ax.quiver(x[pos], y[pos], *-quivers[pos].T, units='xy', scale=2.0, color='r')

    x0, x1, y0, y1 = frame
    # set axis limits / ticks / etc... so we have a nice grid overlay
    ax.set_xlim((x0 - 0.5, num_cols - x1 - 0.5))
    ax.set_ylim((y0 - 0.5, num_rows - y1 - 0.5)[::-1])

    ax.set_xticks(np.arange(x0, num_cols - x1, 1))
    ax.xaxis.set_tick_params(labelsize=5)
    ax.set_yticks(np.arange(y0, num_rows - y1, 1))
    ax.yaxis.set_tick_params(labelsize=5)

    # minor ticks
    ax.set_xticks(np.arange(*ax.get_xlim(), 1), minor=True)
    ax.set_yticks(np.arange(*ax.get_ylim()[::-1], 1), minor=True)

    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
    ax.set_aspect(1)

    tag = f"plots/{title}"
    ax.set_title(title, fontdict={'fontsize': 8, 'fontweight': 'medium'})
    if log_plot:
        plt.savefig(tag.replace(".", "_"))
        assert step is not None
        config.tb.plot(tag, plt, step)
        plt.clf()


def plot_vf(env, vf, title, frame=(0, 0, 0, 0), ax=None, log_plot=False, step=None):
    if ax is None:
        ax = plt.gca()

    x0, x1, y0, y1 = frame
    vf = vf.reshape(env.size, env.size)
    num_cols, num_rows = vf.shape

    tag = f"plots/{title}"
    title = f"{title}_{vf.max():.5f}"

    ax.set_title(title, fontdict={'fontsize': 8, 'fontweight': 'medium'})

    ax.set_xlim((x0 - 0.5, num_cols - x1 - 0.5))
    ax.set_ylim((y0 - 0.5, num_rows - y1 - 0.5)[::-1])

    ax.set_xticks(np.arange(x0, num_cols - x1, 1))
    ax.xaxis.set_tick_params(labelsize=5)
    ax.set_yticks(np.arange(y0, num_rows - y1, 1))
    ax.yaxis.set_tick_params(labelsize=5)

    img = ax.imshow(vf, origin='lower', cmap='viridis')

    ax.set_aspect(1)
    if log_plot:
        assert step is not None
        plt.savefig(tag.replace(".", "_"))
        config.tb.plot(tag, plt, step)
        plt.clf()


def plot_policy_at_state(pi, action_label, title, ax=None, log_plot=False, step=None):
    if ax is None:
        ax = plt.gca()

    pi = pi.flatten()

    for i in range(len(action_label)):
        ax.annotate(f"{pi[i]:.3f}", (i, pi[i]))

    ax.set_xticks(np.arange(len(action_label)), action_label)

    ax.grid(True)

    ax.set_title(title, fontdict={'fontsize': 8, 'fontweight': 'medium'})
    if log_plot:
        plt.savefig(title.replace(".", "_"))
        assert step is not None
        config.tb.plot(title, plt, step)
        plt.clf()


def make_gif(prefix="pi"):
    import glob
    from PIL import Image
    fp_in = os.path.join(config.plot_path, f"{prefix}:*.png")
    fp_out = os.path.join(config.plot_path, f"{prefix}.gif")

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)
