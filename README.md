# README 

Repository containing the code for the paper "Safe  Model-Based  Reinforcement  Learning using Robust Control Barrier Functions". Specifically, an implementation of SAC + Robust Control Barrier Functions (RCBFs) for safe reinforcement learning in two custom environments.

While exploring, an RL agent can take actions that lead the system to unsafe states. Here, we use a differentiable RCBF safety layer that minimially alters (in the least-squares sense) the actions taken by the RL agent to ensure the safety of the agent.

### Robust control barrier functions

As explained in the paper, RCBFs are formulated with respect to differential inclusions that serve to represent disturbed dynamical system (`x_dot \in f(x) + g(x)u + D(x)`). The QP used to ensure the system's safety is given by:
```
u_star(x) = minimize_u ||u||^2 + l ||epsilon||^2
subject to min. h_dot(x, D(x), u, u_RL) > - gamma * h(x) + epsilon
```

In this work, the disturbance set `D` in the differential inclusion is learned via Gaussian Processes (GPs). The underlying library is GPyTorch.  

### Coupling RL & RCBFs to improve training performance

The above is sufficient to ensure the safety of the system, however, we would also like to improve the performance of the learning by letting the RCBF layer guide the training. This is achieved via:
* Using a differentiable version of the safety layer that allows us to backpropagte through the RCBF based Quadratic Program (QP).
* Using the GPs and the dynamics prior to generate synthetic data (model-based RL).

### Other approaches

In addition, the approach is compared against two other frameworks (implemented includedz) in the experiments:
* A vanilla baseline that uses SAC with RCBFs without generating synthetic data nor backproping through the QP (RL loss computed wrt ouput of RL policy).
* A modified approach from ["End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks"](https://ojs.aaai.org/index.php/AAAI/article/view/4213) that replaces their discrete time CBF formulation with RCBFs, but makes use of the supervised learning component to speed up the learning.
