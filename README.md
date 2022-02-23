Code for empirically testing the FMA-PG framework.

This code requires Numpy, Scipy, and Matplotlib to run.

The "current_code" folder contains the main code for running the tabular experiments given in the paper.
- independent code for the CliffWorld and DeepSeaTreasure environments
- code for tabular sMDPO (called sPPO in the code for now), MDPO, TRPO, and sPPO
- code for running ablation experiments (regularized + fixed stepsize, regularized + line search, constrained + line search)
- code for generating sensitivity plots/learning curvse and running scripts

The "notes" folder essentially contains the rough draft of appendix E and F of the main paper.
