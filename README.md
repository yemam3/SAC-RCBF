# README 

An implementation of SAC + Control Barrier Functions (CBFs) for safe-learning in the SafetyGym Environment.

The agent is kept safe throughout learning using a CBF safety layer. CBFs are formulated with respect to control systems 
and are used to guarantee the forward invariance of a set. 

## Basic Components 

### RL component 

The underlying algorithm used here is SAC. 

### CBFs

Traditionally, CBFs require precise knowledge of the dynamics, however, here we use a Robust version of CBFs which keeps the agent safe with respect.
We assume the dynamics take the following form `x_dot = f(x) + g(x)u + D(x)` where `D(x)` is an unknown disturbance term which we
learn using Gaussian Processes (GPs). To deal with higher relative degree CBFs, we use cascaded CBFs.

### GPs for disturbance estimation

We learn the disturbance term `D(x)` using GPs. The underling library is GPyTorch. 

## Sample Efficiency

The objective of the paper is to demonstrate that without making any additional assumptions, we can make the baseline approach
of model-based RL + CBF wrapper significantly more sample efficient. This achieved using two things: model-based rollouts and 
using a differential QP layer. 



