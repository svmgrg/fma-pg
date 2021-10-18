import numpy as np
import matplotlib.pyplot as plt
import pdb

#----------------------------------------------------------------------
# testing code
#----------------------------------------------------------------------
v_star = np.dot(env.calc_v_star(), env.mu)
pi_star = env.calc_pi_star()
v2 = env.calc_vpi(pi_star, FLAG_V_S0=True)

fig, ax = plt.subplots(1, 1)
plot_grid(ax, xlim=7, ylim=7)
plot_policy(ax, pi_star, xlim=7, ylim=7)
plt.axis('equal')
plt.show()

pdb.set_trace()
