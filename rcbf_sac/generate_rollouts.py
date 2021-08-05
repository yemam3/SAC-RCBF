import numpy as np
from copy import deepcopy
from util import euler_to_mat_2d, prCyan, prRed


def generate_model_rollouts(env, memory_model, memory, agent, dynamics_model, goal_loc, k_horizon=1, batch_size=20, warmup=False):

    def policy(observation):

        if warmup and env.action_space:
            action = agent.select_action(observation, dynamics_model, warmup=True)  # Sample action from policy
        else:
            action = agent.select_action(observation, dynamics_model, evaluate=False)  # Sample action from policy

        return action

    # Sample a batch from memory
    if memory.store_t:
        obs_batch, action_batch, reward_batch, next_obs_batch, mask_batch, t_batch, next_t_batch = memory.sample(batch_size=batch_size)
        t_batch_ = deepcopy(t_batch)
    else:
        obs_batch, action_batch, reward_batch, next_obs_batch, mask_batch = memory.sample(batch_size=batch_size)

    obs_batch_ = deepcopy(obs_batch)
    done_batch_ = [False for _ in range(batch_size)]

    for k in range(k_horizon):

        batch_size_ = obs_batch_.shape[0]  # that's because we remove steps where done = True so batch_size shrinks

        action_batch_ = policy(obs_batch_)
        state_batch_ = dynamics_model.get_state(obs_batch_)
        if memory.store_t:
            next_state_mu_, next_state_std_, next_t_batch_ = dynamics_model.predict_next_state(state_batch_, action_batch_, t_batch=t_batch)
        else:
            next_state_mu_, next_state_std_ = dynamics_model.predict_next_state(state_batch_, action_batch_)
        next_state_batch_ = np.random.normal(next_state_mu_, next_state_std_)
        next_obs_batch_ = dynamics_model.get_obs(next_state_batch_)

        if env.dynamics_mode == 'Unicycle':

            # Construct Next Observation from State
            dist2goal_prev = -np.log(obs_batch_[:, -1])
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

        elif env.dynamics_mode == 'SimulatedCars':

            # Compute Reward
            car_4_pos = next_obs_batch_[:, 6]  # car's 4 position
            car_4_vel = next_obs_batch_[:, 7]  # car's 4 velocity
            reward_batch_ = -np.abs(car_4_vel) * np.abs(action_batch_) * (action_batch_ > 0) / env.max_episode_steps
            if (next_obs_batch_[:, 4] - car_4_pos) < 2.99:  # How far is car 3?
                reward_batch_ -= np.abs(500 / (next_obs_batch_[:, 4] - car_4_pos))
            if (car_4_pos - next_obs_batch_[8]) < 2.99:  # How far is car 4?
                reward_batch_ -= np.abs(500 / (car_4_pos - next_obs_batch_[:, 8]))

            # Compute Done
            done_batch_ = next_t_batch_ >= env.max_episode_steps * env.dt  # done?

        else:
            raise Exception('Environment/Dynamics mode {} not Recognized!'.format(env.dynamics_mode))

        if memory.store_t:
            memory_model.batch_push(obs_batch_, action_batch_, reward_batch_, next_obs_batch_, mask_batch_, t_batch_, next_t_batch_)  # Append transition to memory
            t_batch_ = deepcopy(next_t_batch_)
        else:
            memory_model.batch_push(obs_batch_, action_batch_, reward_batch_, next_obs_batch_, mask_batch_)  # Append transition to memory

        # Update Current Observation Batch
        obs_batch_ = deepcopy(next_obs_batch_)

        # Delete Done Trajectories
        # prCyan('batch_size before omission = {}'.format(obs_batch_.shape))
        if np.sum(done_batch_) > 0:
            print('reached_goal = {}'.format(done_batch_))
            obs_batch_ = np.delete(obs_batch_, done_batch_ > 0, axis=0)
        # prCyan('batch_size after omission = {}, #dones = {}'.format(obs_batch_.shape, np.sum(done_batch_)))

    return memory_model
