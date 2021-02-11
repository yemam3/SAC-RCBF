from safety_gym.envs.engine import Engine
import gym
import numpy as np
from utils.util import mat_to_euler_2d

"""
More info can be obtained here: 
https://github.com/openai/safety-gym/blob/master/safety_gym/envs/engine.py
"""

DT = 0.002  # timestep


class ObsWrapper(gym.ObservationWrapper):

    def __init__(self, env, hazards_locations, hazards_radius):
        """

        Parameters
        ----------
        env : gym.env
            Safety gym environment
        hazards_locations : list
            List of hazard-xy positions where each item is a list of length 2
        hazards_radius : float
            Radius of the hazards
        """

        super(ObsWrapper, self).__init__(env)
        self.action_space = env.action_space
        self.dt = DT
        self.hazards_locations = hazards_locations
        self.hazards_radius = hazards_radius

    def observation(self, obs):
        """Organize the observation to understand what's going on

        Parameters
        ----------
        obs : ndarray

        Returns
        -------
        new_obs : ndarray
            The new observation with the following structure:
            [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, dist2goal]

        """

        theta = mat_to_euler_2d(self.env.data.get_body_xmat('robot'))

        robot_state = np.array([obs[0], obs[1], np.cos(theta), np.sin(theta)])
        new_obs = np.append(robot_state, obs[3:])
        return new_obs


def build_env(args):
    """Build our own env using the Engine."""

    # Hazards (danger areas)
    hazards_locations = [[-1., 0.], [1., 0.], [0., -1.], [0., 1.]]
    robot_locations = [[0., 0.]]
    hazards_radius = 0.6

    config = {
        'robot_base': args.robot_xml,
        'robot_locations': robot_locations,
        'task': 'goal',
        'observe_com': True,  # observe center of mass of robot xyz
        'observe_qpos': False,
        'observe_qvel': False,
        'observe_goal_comp': True,  # observe goal_compass
        'observe_goal_dist': True,  # observe
        'hazards_num': len(hazards_locations),
        'hazards_locations': hazards_locations,
        'hazards_size': hazards_radius,
        'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
        'observe_sensors': False,  # whether measurements from `sensors_obs` should be in the observation
        'observe_goal_lidar': False,  # if goals should be observed by LIDAR
        'observe_box_lidar': False,  # if boxes should be observed by LIDAR
        'observe_hazards': False,  # Observe the vector from agent to hazards (LIDAR like)
        'observe_vases': False,  # Observe the vector from agent to vases (LIDAR like)
        'constrain_hazards': True,
        'lidar_max_dist': 3,
        'lidar_num_bins': 16,
        'vases_num': 0,
        'build_resample': False
    }

    env = ObsWrapper(Engine(config), hazards_locations, hazards_radius)

    return env
