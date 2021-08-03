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
        self.action_space = spaces.Box(low=-100.0, high=100.0, shape=(1,))
        self.observation_space = spaces.Box(low=-1e10, high=1e10, shape=(10,))
        self._max_episode_steps = 80
        self.dt = 0.05

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

        # Actions (acceleration)
        vels_des = 30.0 * np.ones(5)  # Desired velocities
        vels_des[0] -= 10*np.sin(0.2*self.t)
        accels = self.kp * (vels_des - vels)
        p_diff = np.append(1.0e10, -np.diff(pos))
        mask = p_diff < 6.0
        accels[mask] -= self.k_brake * p_diff[mask]
        accels[3] = 0.0  # Car 4 is controlled directly

        # Determine action of each car
        f_x = np.zeros(10)
        g_x = np.zeros(10)

        f_x[::2] = vels  # Derivatives of positions are velocities
        f_x[1::2] = accels  # Derivatives of velocities are acceleration
        g_x[7] = 1.0  # Car 4's acceleration (idx = 2*4 - 1) is the control input

        self.state += self.dt * (f_x + g_x * action)

        self.t = self.t + 0.05  # time

        self.episode_step += 1  # steps in episode

        done = self.episode_step >= self._max_episode_steps # done?

        info = {}

        return self._get_obs(), self._get_reward(action), done, info

    def _get_reward(self, action):

        car_4_pos = self.state[6]  # car's 4 position
        car_4_vel = self.state[7]  # car's 4 velocity

        r = -np.abs(car_4_vel) * action * (action > 0)

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

    def func_to_vectorize(x, y, dx, dy, scaling=0.01):
        plt.arrow(x, y, dx*scaling, dy*scaling, fc="k", ec="k", head_width=0.06, head_length=0.1)

    vectorized_arrow_drawing = np.vectorize(func_to_vectorize)

    env = SimulatedCarsEnv()
    obs = env.reset()
    done = False

    # Plot initial state
    p_pos = plt.plot(obs[::2], np.zeros(5), 'bo')[0]
    p_vel = plt.quiver(obs[::2], np.zeros(5), obs[1::2], np.zeros(5))

    while not done:
        # Plot current state
        pos = obs[::2]
        p_pos.set_xdata(obs[::2])
        p_vel.XY[:, 0] = obs[::2]
        p_vel.set_UVC(obs[1::2], np.zeros(5))
        # Take Action and get next state
        random_action = 2.0 * (np.random.random() - 0.5)
        obs, reward, done, info = env.step(random_action)
        plt.xlim([pos[-1] - 5.0, pos[0] + 5.0])
        plt.pause(0.1)

    plt.show()