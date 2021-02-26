import numpy as np
from copy import deepcopy
from util import euler_to_mat_2d, get_wrapped_policy


def generate_model_rollouts(env, memory_model, memory, agent, cbf_wrapper, dynamics_model, goal_loc, compensator, k_horizon=1,
                            batch_size=20, warmup=False):

    wrapped_policy = get_wrapped_policy(agent, cbf_wrapper, dynamics_model, compensator=compensator, warmup=warmup, action_space=env.action_space)

    # Sample a batch from memory
    obs_batch, action_batch, reward_batch, next_obs_batch, mask_batch = memory.sample(batch_size=batch_size)

    # TODO: Speed this up
    for i, init_obs in enumerate(obs_batch):

        done = not mask_batch[i]
        obs = init_obs

        for k in range(k_horizon):

            if done or reward_batch[i] < -5.0:
                break

            wrapped_action = wrapped_policy(obs)
            state = dynamics_model.get_state(obs)

            next_state_mu, next_state_std = dynamics_model.predict_next_state(state, wrapped_action)
            next_state = np.random.normal(next_state_mu, next_state_std)

            # TODO: Learn rest of obs using NNs instead of currently hand-tailored
            next_obs = dynamics_model.get_obs(next_state)
            dist2goal_prev = -np.log(obs[-1])
            goal_rel = goal_loc - next_obs[:2]
            dist2goal = np.linalg.norm(goal_rel)
            # generate compass
            compass = np.matmul(goal_rel, euler_to_mat_2d(next_state[2]))
            compass /= np.sqrt(np.sum(np.square(compass))) + 0.001
            next_obs = np.hstack((next_obs, compass, np.exp(-dist2goal)))

            # TODO: what is the reward function? What is the mask?
            mask = True  # never assume terminal

            # TODO: Specify those in build_env.py and pass env to this function
            # reward = (self.last_dist_goal - dist_goal) * self.reward_distance
            # reward += self.reward_goal * (self.dist_goal() <= self.goal_size)
            goal_size = 0.3
            reward_goal = 1.0
            reward_distance = 1.0
            reward = (dist2goal_prev - dist2goal) * reward_distance + (dist2goal <= goal_size) * reward_goal

            memory_model.push(obs, wrapped_action, reward, next_obs, mask)  # Append transition to memory

            obs = deepcopy(next_obs)

    return memory_model