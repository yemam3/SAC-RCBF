import numpy as np
import gym
from gym import spaces


class SimulatedCarsEnv(gym.Env):
    """Simulated Car Following Env, almost identical to https://github.com/rcheng805/RL-CBF/blob/master/car/DDPG/car_simulator.py
    Front <- Car 1 <- Car 2 <- Car 3 <- Car 4 (controlled) <- Car 5
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):

        super(SimulatedCarsEnv, self).__init__()

        self.dynamics_mode = 'SimulatedCars'
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.safe_action_space = spaces.Box(low=-20.0, high=20.0, shape=(1,))
        self.observation_space = spaces.Box(low=-1e10, high=1e10, shape=(10,))
        self.max_episode_steps = 160
        self.dt = 0.02

        # Gains
        self.kp = 4.0
        self.k_brake = 20.0

        self.state = None  # State [x_1 v_1 ... x_5 v_5]
        self.t = 0  # Time
        self.episode_step = 0  # Episode Step

        self.reset()

    def step(self, action):
        """Organize the observation to understand what's going on

        Parameters
        ----------
        action : ndarray
                Action that the agent takes in the environment

        Returns
        -------
        new_obs : ndarray
          The new observation with the following structure:
          [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, dist2goal]

        """

        # Current State
        pos = self.state[::2]
        vels = self.state[1::2]

        # Actions (accelerations of Cars 1 to 5)
        vels_des = 30.0 * np.ones(5)  # Desired velocities
        vels_des[0] -= 10*np.sin(0.2*self.t)
        accels = self.kp * (vels_des - vels)
        accels[1] -= self.k_brake * (pos[0] - pos[1]) * ((pos[0] - pos[1]) < 6.0)
        accels[2] -= self.k_brake * (pos[1] - pos[2]) * ((pos[1] - pos[2]) < 6.0)
        accels[3] = 0.0  # Car 4's acceleration is controlled directly
        accels[4] -= self.k_brake * (pos[2] - pos[4]) * ((pos[2] - pos[4]) < 12.0)

        # Determine action of each car
        f_x = np.zeros(10)
        g_x = np.zeros(10)

        f_x[::2] = vels  # Derivatives of positions are velocities
        f_x[1::2] = accels  # Derivatives of velocities are acceleration
        g_x[7] = 50.0  # Car 4's acceleration (idx = 2*4 - 1) is the control input

        self.state += self.dt * (f_x + g_x * action)

        self.t = self.t + 0.05  # time

        self.episode_step += 1  # steps in episode

        done = self.episode_step >= self.max_episode_steps  # done?

        info = {}

        return self._get_obs(), self._get_reward(action[0]), done, info

    def _get_reward(self, action):

        car_4_pos = self.state[6]  # car's 4 position
        car_4_vel = self.state[7]  # car's 4 velocity

        r = - np.abs(car_4_vel) * np.abs(action) * (action > 0) / self.max_episode_steps

        if (self.state[4] - car_4_pos) < 2.99:  # How far is car 3?
            r -= np.abs(500 / (self.state[4] - car_4_pos))

        if (car_4_pos - self.state[8]) < 2.99:  # How far is car 4?
            r -= np.abs(500 / (car_4_pos - self.state[8]))

        return r

    def reset(self):
        """ Reset the state of the environment to an initial state.

        Returns
        -------
        observation : ndarray
            Next observation.
        """

        self.t = 0
        self.state = np.zeros(10)  # first col is pos, 2nd is vel
        self.state[::2] = [34.0, 28.0, 22.0, 16.0, 10.0]  # initial positions
        self.state[1::2] = 30.0  # initial velocities

        self.episode_step = 0

        return self._get_obs()


    def render(self, mode='human', close=False):
        """Render the environment to the screen

        Parameters
        ----------
        mode : str
        close : bool

        Returns
        -------

        """

        print('Ep_step = {}, \tState = {}'.format(self.episode_step, self.state))

    def _get_obs(self):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------

        Returns
        -------
        observation : ndarray
          Observation: [car_1_x, car_1_v, car_1_a, ...]
        """

        return np.ravel(self.state)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from archive.cbf_cascade import CascadeCBFLayer
    from dynamics import DynamicsModel
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="SafetyGym", help='Either SafetyGym or Unicycle.')
    parser.add_argument('--gp_model_size', default=2000, type=int, help='gp')
    parser.add_argument('--k_d', default=2.0, type=float)
    parser.add_argument('--gamma_b', default=50, type=float)
    parser.add_argument('--cuda', action="store_true", help='run on CUDA (default: False)')
    args = parser.parse_args()

    env = SimulatedCarsEnv()
    dynamics_model = DynamicsModel(env, args)
    cbf_wrapper = CascadeCBFLayer(env, gamma_b=args.gamma_b, k_d=args.k_d)

    obs = env.reset()
    done = False
    episode_reward = 0

    # Plot initial state
    p_pos = plt.scatter(obs[::2], np.zeros(5), marker='s', s=13*60, cmap='Accent', c=list(range(5)))
    p_vel = plt.quiver(obs[::2], np.zeros(5), obs[1::2], np.zeros(5))

    while not done:
        # Plot current state
        pos = obs[::2]
        p_pos.set_offsets(np.c_[pos, np.zeros(pos.shape)])
        p_vel.XY[:, 0] = obs[::2]
        p_vel.set_UVC(obs[1::2], np.zeros(5))
        # Take Action and get next state
        random_action = env.action_space.sample()
        disturb_mean, disturb_std = dynamics_model.predict_disturbance(obs)
        action_safe = cbf_wrapper.get_u_safe(random_action, obs, disturb_mean, disturb_std)
        obs, reward, done, info = env.step(random_action + action_safe)
        plt.xlim([pos[-1] - 5.0, pos[0] + 5.0])
        plt.pause(0.01)
        episode_reward += reward


        print('episode_reward = {}'.format(episode_reward))
    plt.show()
