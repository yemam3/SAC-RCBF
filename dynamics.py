import numpy as np
import argparse
import torch
from utils.gp_model import GPyDisturbanceEstimator
from copy import deepcopy

DYNAMICS_MODE = {'unicycle': {'n_s': 3, 'n_u': 2}}
MAX_STD = {'unicycle': [1e-3, 1e-3, 1e-3]}


class DynamicsModel:

    def __init__(self, env, args, max_history_count=1000):
        """Constructor of DynamicsModel.

        Parameters
        ----------
        env : gym.env
            Gym environment.
        """

        self.dynamics_mode = args.dynamics_mode
        self.env = env
        # Get Dynamics
        self.get_f, self.get_g = self.get_dynamics()
        self.n_s = DYNAMICS_MODE[self.dynamics_mode]['n_s']
        self.n_u = DYNAMICS_MODE[self.dynamics_mode]['n_u']

        # Keep Disturbance History to estimate it using GPs
        self.disturb_estimators = None
        self.disturbance_history = dict()
        self.history_counter = 0  # keeping only 1000 points in the buffer
        self.max_history_count = max_history_count  # How many points we want to have in the GP
        self.disturbance_history['state'] = np.zeros((self.max_history_count, self.n_s))
        self.disturbance_history['disturbance'] = np.zeros((self.max_history_count, self.n_s))
        self.train_x = None  # x-data used to fit the last GP models
        self.train_y = None  # y-data used to fit the last GP models

    def predict_next_state(self, state, u, use_gps=True):
        """Given the current state and action, this function predicts the next state.

        Parameters
        ----------
        state : ndarray
            State
        u : ndarray
            Action
        use_gps : bool, optional
            Use GPs to return mean and var

        Returns
        -------
        next_state : ndarray
            Next state
        """

        # Start with our prior x' = f(x) + g(x)u
        next_state = self.get_f(state) + self.get_g(state) @ u

        if use_gps:  # if we want estimate the disturbance, let's do it!
            pred_mean, pred_std = self.predict_disturbance(state)
            next_state += pred_mean
        else:
            pred_std = None

        return next_state, pred_std

    def predict_next_obs(self, state, u):
        """Predicts the next observation given the state and u. Note that this only predicts the mean next observation.

        Parameters
        ----------
        state : ndarray
        u : ndarray

        Returns
        -------
        next_obs : ndarray
            Next observation
        """

        next_state, _, _ = self.predict_next_state(state, u)
        next_obs = self.get_obs(next_state)
        return next_obs

    def get_dynamics(self):
        """Get affine CBFs for a given environment.

        Parameters
        ----------
        env_name : str
            String specifying the name of the environment.

        Returns
        -------
        get_f : callable
                Drift dynamics of the discrete system x' = f(x) + g(x)u
        get_g : callable
                Control dynamics of the discrete system x' = f(x) + g(x)u
        """

        dt = self.env.dt

        if self.dynamics_mode == 'unicycle':

            def get_f(state):
                f_x = state
                return f_x

            def get_g(state):
                theta = state[2]
                g_x = dt * np.array([[np.cos(theta), 0],
                                     [np.sin(theta), 0],
                                     [            0, 1]])
                return g_x

            return get_f, get_g

    def get_state(self, obs):
        """Given the observation, this function does the pre-processing necessary and returns the state.

        Parameters
        ----------
        obs : ndarray
            Environment observation.

        Returns
        -------
        state : ndarray
            State of the system.

        """

        if self.dynamics_mode == 'unicycle':
            theta = np.arctan2(obs[3], obs[2])
            state = np.array([obs[0], obs[1], theta])
            return state

    def get_obs(self, state):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------
        state : ndarray
            Environment state.

        Returns
        -------
        state : ndarray
            State of the system.

        """

        if self.dynamics_mode == 'unicycle':
            obs = np.array([state[0], state[1], np.cos(state[0]), np.sin(state[0])])
            return obs

    def append_transition(self, state, u, next_state):
        """Estimates the disturbance from the current dynamics transition and adds it to buffer.

        Parameters
        ----------
        state : ndarray, shape(n_s,)
        u : ndarray, shape(n_u,)
        next_state : ndarray, shape(n_s,)

        Returns
        -------

        """

        next_state_prior_pred, _ = self.predict_next_state(state, u, use_gps=False)
        disturbance = next_state - next_state_prior_pred

        # Append new data point (state, disturbance) to our dataset
        self.disturbance_history['state'][self.history_counter % self.max_history_count] = state
        self.disturbance_history['disturbance'][self.history_counter % self.max_history_count] = disturbance

        # Increment how many data points we have
        self.history_counter += 1

        # Update GP models every max_history_count data points
        if self.history_counter % (self.max_history_count/2) == 0:
            self.fit_gp_model()

    def fit_gp_model(self, training_iter=50):
        """

        Parameters
        ----------
        training_iter : int
            Number of training iterations for GP model.

        Returns
        -------

        """

        if self.history_counter < self.max_history_count:  # didn't fill the buffer yet
            train_x = self.disturbance_history['state'][:self.history_counter]
            train_y = self.disturbance_history['disturbance'][:self.history_counter]
        else:  # buffer filled, use all the data points
            train_x = self.disturbance_history['state']
            train_y = self.disturbance_history['disturbance']

        self.disturb_estimators = []
        for i in range(self.n_s):
            self.disturb_estimators.append(GPyDisturbanceEstimator(train_x, train_y[:, i]))
            self.disturb_estimators[i].train(training_iter)

        # track the data I last used to fit the GPs for saving purposes (need it to initialize before loading weights)
        self.train_x = train_x
        self.train_y = train_y

    def predict_disturbance(self, test_x):
        """

        Parameters
        ----------
        test_x : ndarray, shape(n_test, n_s)

        Returns
        -------
        means: ndarray
            Prediction means -- shape(n_test, n_s)
        vars: ndarray
            Prediction variances -- shape(n_test, n_s)
        """

        if len(test_x.shape) == 1:
            test_x = np.expand_dims(test_x, axis=0)

        means = np.zeros(test_x.shape)
        f_std = np.zeros(test_x.shape)  # standard deviation

        if self.disturb_estimators:
            for i in range(self.n_s):
                prediction_ = self.disturb_estimators[i].predict(test_x)
                means[:, i] = prediction_['mean']
                f_std[:, i] = np.sqrt(prediction_['f_var'])

        else:  # zero-mean, max_sigma prior
            f_std = np.ones(test_x.shape)
            for i in range(self.n_s):
                f_std[:, i] *= MAX_STD[self.dynamics_mode][i]

        return means.squeeze(), f_std.squeeze()

    def load_disturbance_models(self, output):

        if output is None:
            return

        self.disturb_estimators = []

        try:
            weights = torch.load('{}/gp_models.pkl'.format(output))
            train_x = torch.load('{}/gp_models_train_x.pkl'.format(output))
            train_y = torch.load('{}/gp_models_train_y.pkl'.format(output))
            for i in range(self.n_s):
                self.disturb_estimators.append(GPyDisturbanceEstimator(train_x, train_y[:, i]))
                self.disturb_estimators[i].model.load_state_dict(weights[i])
        except:
            raise Exception('Could not load GP models from {}'.format(output))

    def save_disturbance_models(self, output):

        if not self.disturb_estimators or self.train_x is None or self.train_y is None:
            return
        weights = []
        for i in range(len(self.disturb_estimators)):
            weights.append(self.disturb_estimators[i].model.state_dict())
        torch.save(weights, '{}/gp_models.pkl'.format(output))
        # Also save data used to fit model (needed for initializing the model before loading weights)
        torch.save(self.train_x, '{}/gp_models_train_x.pkl'.format(output))
        torch.save(self.train_y, '{}/gp_models_train_y.pkl'.format(output))

    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)

