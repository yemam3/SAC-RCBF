import numpy as np
from copy import deepcopy
from util import euler_to_mat_2d, prCyan


def generate_model_rollouts(env, memory_model, memory, agent, dynamics_model, goal_loc, k_horizon=1, batch_size=20, warmup=False):

    def policy(observation):

        if warmup and env.action_space:
            action = agent.select_action(observation, dynamics_model, warmup=True)  # Sample action from policy
        else:
            action = agent.select_action(observation, dynamics_model, evaluate=False)  # Sample action from policy

        return action

    # Sample a batch from memory
    obs_batch, action_batch, reward_batch, next_obs_batch, mask_batch = memory.sample(batch_size=batch_size)
    #
    obs_batch_ = deepcopy(obs_batch)
    done_batch_ = [False for _ in range(batch_size)]

    for k in range(k_horizon):

        batch_size_ = obs_batch_.shape[0]  # that's because we remove steps where done = True so batch_size shrinks

        action_batch_ = policy(obs_batch_)
        state_batch_ = dynamics_model.get_state(obs_batch_)
        next_state_mu_, next_state_std_ = dynamics_model.predict_next_state(state_batch_, action_batch_)
        next_state_batch_ = np.random.normal(next_state_mu_, next_state_std_)

        # Construct Next Observation from State
        next_obs_batch_ = dynamics_model.get_obs(next_state_batch_)
        dist2goal_prev = -np.log(next_obs_batch_[:, -1])
        goal_rel = goal_loc - next_obs_batch_[:, :2]
        dist2goal = np.linalg.norm(goal_rel, axis=1)
        assert dist2goal.shape == (batch_size_,), 'dist2goal should be a vector of size (batch_size,), got {} instead'.format(dist2goal.shape)
        # generate compass
        compass = np.matmul(np.expand_dims(goal_rel, 1), euler_to_mat_2d(next_state_batch_[:, 2])).squeeze(1)
        compass /= np.sqrt(np.sum(np.square(compass), axis=1, keepdims=True)) + 0.001
        next_obs_batch_ = np.hstack((next_obs_batch_, compass, np.expand_dims(np.exp(-dist2goal), axis=-1)))

        # Compute Reward
        goal_size = 0.3
        reward_goal = 1.0
        reward_distance = 1.0
        reward_batch_ = (dist2goal_prev - dist2goal) * reward_distance + (dist2goal <= goal_size) * reward_goal
        # Compute Done
        reached_goal = dist2goal <= goal_size
        reward_batch_ += reward_goal * reached_goal
        done_batch_ = reached_goal
        mask_batch_ = np.invert(done_batch_)

        for i in range(batch_size_):
            memory_model.push(obs_batch_[i], action_batch_[i], reward_batch_[i], next_obs_batch_[i], mask_batch_[i])  # Append transition to memory

        # Update Current Observation Batch
        obs_batch_ = deepcopy(next_obs_batch_)

        # Delete Done Trajectories
        # prCyan('batch_size before omission = {}'.format(obs_batch_.shape))
        if np.sum(done_batch_) > 0:
            obs_batch_[done_batch_ > 0] = []
        # prCyan('batch_size after omission = {}, #dones = {}'.format(obs_batch_.shape, np.sum(done_batch_)))

    return memory_model
