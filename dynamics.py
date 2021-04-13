import numpy as np
import torch
from gp_model import GPyDisturbanceEstimator
from util import to_tensor, to_numpy

"""
This file contains the DynamicsModel class. Depending on the environment selected, it contains the dynamics
priors and learns the remaining disturbance term using GPs. The system is described as follows: 
                        x_dot ∈ f(x) + g(x)u + D(x), 
where x is the state, f and g are the drift and control dynamics respectively, and D(x) is the learned disturbance set. 

A few things to note: 
    - The prior depends on the dynamics. This is hard-coded as of now. In the future, one direction to take would be to 
    get a few episodes in the env to determine the affine prior first using least-squares or something similar.
    - The state is not the same as the observation and typically requires some pre-processing. These functions are supplied
    here which are also, unfortunately, hard-coded as of now. 
    - The functions here are sometimes used in batch-form or single point form, and sometimes the inputs are torch tensors 
    and other times are numpy arrays. These are things to be mindful of, and output format is always same as input format.
"""


DYNAMICS_MODE = {'Unicycle': {'n_s': 3, 'n_u': 2},   # state = [x y θ]
                 'SafetyGym_point': {'n_s': 5, 'n_u': 2}}  # state = [x y θ v ω]
MAX_STD = {'Unicycle': [2e-1, 2e-1, 2e-1], 'SafetyGym_point': [0, 0, 0, 50, 200.]}


class DynamicsModel:

    def __init__(self, env, args):
        """Constructor of DynamicsModel.

        Parameters
        ----------
        env : gym.env
            Gym environment.
        """

        self.env = env
        # Get Dynamics
        self.get_f, self.get_g = self.get_dynamics()
        self.n_s = DYNAMICS_MODE[self.env.dynamics_mode]['n_s']
        self.n_u = DYNAMICS_MODE[self.env.dynamics_mode]['n_u']

        # Keep Disturbance History to estimate it using GPs
        self.disturb_estimators = None
        self.disturbance_history = dict()
        self.history_counter = 0  # keeping only 1000 points in the buffer
        self.max_history_count = args.gp_model_size  # How many points we want to have in the GP
        self.disturbance_history['state'] = np.zeros((self.max_history_count, self.n_s))
        self.disturbance_history['disturbance'] = np.zeros((self.max_history_count, self.n_s))
        self.train_x = None  # x-data used to fit the last GP models
        self.train_y = None  # y-data used to fit the last GP models

        # Point Robot specific dynamics (approx using unicycle + look-ahead)
        self.l_p = args.l_p

        self.device = torch.device("cuda" if args.cuda else "cpu")

    def predict_next_state(self, state_batch, u_batch, use_gps=True):
        """Given the current state and action, this function predicts the next state.

        Parameters
        ----------
        state_batch : ndarray
            State
        u_batch : ndarray
            Action
        use_gps : bool, optional
            Use GPs to return mean and var

        Returns
        -------
        next_state : ndarray
            Next state
        """

        expand_dims = len(state_batch.shape) == 1
        if expand_dims:
            state_batch = np.expand_dims(state_batch, dim=0)

        # Start with our prior for continuous time system x' = f(x) + g(x)u
        next_state_batch = state_batch + self.env.dt * (self.get_f(state_batch) + (self.get_g(state_batch) @ np.expand_dims(u_batch, -1)).squeeze(-1))

        if use_gps:  # if we want estimate the disturbance, let's do it!
            pred_mean, pred_std = self.predict_disturbance(state_batch)
            next_state_batch += self.env.dt * pred_mean
        else:
            pred_std = None

        if expand_dims:
            next_state_batch = next_state_batch.squeeze(0)
            if pred_std is not None:
                pred_std = pred_std.squeeze(0)

        return next_state_batch, self.env.dt * pred_std

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

        Returns
        -------
        get_f : callable
                Drift dynamics of the continuous system x' = f(x) + g(x)u
        get_g : callable
                Control dynamics of the continuous system x' = f(x) + g(x)u
        """

        if self.env.dynamics_mode == 'Unicycle':

            def get_f(state_batch):
                f_x = np.zeros(state_batch.shape)
                return f_x

            def get_g(state_batch):
                theta = state_batch[:, 2]
                g_x = np.zeros((state_batch.shape[0], 3, 2))
                g_x[:, 0, 0] = np.cos(theta)
                g_x[:, 1, 0] = np.sin(theta)
                g_x[:, 2, 1] = 1.0
                return g_x

        elif self.env.dynamics_mode == 'SafetyGym_point':

            def get_f(state_batch):
                f_x = np.zeros(state_batch.shape)
                f_x[:, 0] = 9.*state_batch[:, 3] * np.cos(state_batch[:, 2])  # x_dot = v*cos(θ)
                f_x[:, 1] = 9.*state_batch[:, 3] * np.sin(state_batch[:, 2])  # y_dot = v*sin(θ)
                f_x[:, 2] = 5.5*state_batch[:, 4]  # θ_dot = ω
                f_x[:, 3] = -20.*state_batch[:, 3]  # v_dot = u^ - damp_coeff * v
                f_x[:, 4] = -500.*state_batch[:, 4]  # ω_dot = u^ω - damp_coeff * ω
                return f_x

            def get_g(state_batch):
                g_x = np.zeros((state_batch.shape[0], 5, 2))
                g_x[:, 3, 0] = 40.0  # v_dot = u^v - damp_coeff * ω
                g_x[:, 4, 1] = 1520.0  # ω_dot = u^ω - damp_coeff * ω
                return g_x


        else:
            raise Exception('Unknown Dynamics mode.')

        return get_f, get_g

    def get_state(self, obs):
        """Given the observation, this function does the pre-processing necessary and returns the state.

        Parameters
        ----------
        obs_batch : ndarray or torch.tensor
            Environment observation.

        Returns
        -------
        state_batch : ndarray or torch.tensor
            State of the system.

        """

        expand_dims = len(obs.shape) == 1
        is_tensor = torch.is_tensor(obs)

        if is_tensor:
            dtype = obs.dtype
            device = obs.device
            obs = to_numpy(obs)

        if expand_dims:
            obs = np.expand_dims(obs, 0)

        if self.env.dynamics_mode == 'Unicycle':
            theta = np.arctan2(obs[:, 3], obs[:, 2])
            state_batch = np.zeros((obs.shape[0], 3))
            state_batch[:, 0] = obs[:, 0]
            state_batch[:, 1] = obs[:, 1]
            state_batch[:, 2] = theta
        elif self.env.dynamics_mode == 'SafetyGym_point':
            theta = np.arctan2(obs[:, 3], obs[:, 2])
            state_batch = np.zeros((obs.shape[0], 5))
            state_batch[:, 0] = obs[:, 0]  # x
            state_batch[:, 1] = obs[:, 1]  # y
            state_batch[:, 2] = theta
            state_batch[:, 3] = obs[:, 4]  # v
            state_batch[:, 4] = obs[:, 5]  # omega
        else:
            raise Exception('Unknown dynamics')

        if expand_dims:
            state_batch = state_batch.squeeze(0)

        return to_tensor(state_batch, dtype, device) if is_tensor else state_batch

    def get_obs(self, state_batch):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------
        state : ndarray
            Environment state batch of shape (batch_size, n_s)

        Returns
        -------
        obs : ndarray
          Observation batch of shape (batch_size, n_o)

        """

        if self.env.dynamics_mode == 'Unicycle':
            obs = np.zeros((state_batch.shape[0], 4))
            obs[:, 0] = state_batch[:, 0]
            obs[:, 1] = state_batch[:, 1]
            obs[:, 2] = np.cos(state_batch[:, 2])
            obs[:, 3] = np.sin(state_batch[:, 2])
        elif self.env.dynamics_mode == 'SafetyGym_point':
            obs = np.zeros((state_batch.shape[0], 6))
            obs[:, 0] = state_batch[:, 0]
            obs[:, 1] = state_batch[:, 1]
            obs[:, 2] = np.cos(state_batch[:, 2])
            obs[:, 3] = np.sin(state_batch[:, 2])
            obs[:, 4] = state_batch[:, 3]
            obs[:, 5] = state_batch[:, 4]
        else:
            raise Exception('Unknown dynamics')
        return obs

    def append_transition(self, state_batch, u_batch, next_state_batch):
        """Estimates the disturbance from the current dynamics transition and adds it to buffer.

        Parameters
        ----------
        state_batch : ndarray
            shape (n_s,) or (batch_size, n_s)
        u_batch : ndarray
            shape (n_u,) or (batch_size, n_u)
        next_state_batch : ndarray
            shape (n_s,) or (batch_size, n_s)

        Returns
        -------

        """

        expand_dims = len(state_batch.shape) == 1

        if expand_dims:
            state_batch = np.expand_dims(state_batch, 0)
            next_state_batch = np.expand_dims(next_state_batch, 0)
            u_batch = np.expand_dims(u_batch, 0)

        u_batch = np.expand_dims(u_batch, -1)  # for broadcasting batch matrix multiplication purposes

        disturbance_batch = (next_state_batch - state_batch - self.env.dt * (self.get_f(state_batch) + (self.get_g(state_batch) @ u_batch).squeeze(-1))) / self.env.dt

        # Append new data point (state, disturbance) to our dataset
        for i in range(state_batch.shape[0]):

            self.disturbance_history['state'][self.history_counter % self.max_history_count] = state_batch[i]
            self.disturbance_history['disturbance'][self.history_counter % self.max_history_count] = disturbance_batch[i]

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

        # Normalize Data
        train_x_std = np.std(train_x, axis=0)
        train_x_normalized = train_x / (train_x_std + 1e-8)
        train_y_std = np.std(train_y, axis=0)
        train_y_normalized = train_y / (train_y_std + 1e-8)

        self.disturb_estimators = []
        for i in range(self.n_s):
            # self.disturb_estimators.append(GPyDisturbanceEstimator(train_x, train_y[:, i]))
            self.disturb_estimators.append(GPyDisturbanceEstimator(train_x_normalized, train_y_normalized[:, i], device=self.device))
            self.disturb_estimators[i].train(training_iter)

        # track the data I last used to fit the GPs for saving purposes (need it to initialize before loading weights)
        self.train_x = train_x
        self.train_y = train_y

    def predict_disturbance(self, test_x):
        """Predict the disturbance at the queried states using the GP models.

        Parameters
        ----------
        test_x : ndarray or torch.tensor
                shape(n_test, n_s)
        Returns
        -------
        means: ndarray or torch.tensor
            Prediction means -- shape(n_test, n_s)
        vars: ndarray or torch.tensor
            Prediction variances -- shape(n_test, n_s)
        """

        is_tensor = torch.is_tensor(test_x)

        if is_tensor:
            dtype = test_x.dtype
            device = test_x.device
            test_x = to_numpy(test_x)

        expand_dims = len(test_x.shape) == 1
        if expand_dims:
            test_x = np.expand_dims(test_x, axis=0)

        means = np.zeros(test_x.shape)
        f_std = np.zeros(test_x.shape)  # standard deviation

        if self.disturb_estimators:
            # Normalize
            train_x_std = np.std(self.train_x, axis=0)
            train_y_std = np.std(self.train_y, axis=0)
            test_x = test_x / train_x_std
            for i in range(self.n_s):
                prediction_ = self.disturb_estimators[i].predict(test_x)
                means[:, i] = prediction_['mean'] * (train_y_std[i] + 1e-8)
                f_std[:, i] = np.sqrt(prediction_['f_var']) * (train_y_std[i] + 1e-8)

        else:  # zero-mean, max_sigma prior
            f_std = np.ones(test_x.shape)
            for i in range(self.n_s):
                f_std[:, i] *= MAX_STD[self.env.dynamics_mode][i]

        if expand_dims:
            means = means.squeeze(0)
            f_std = f_std.squeeze(0)

        return (to_tensor(means, dtype, device), to_tensor(f_std, dtype, device)) if is_tensor else (means, f_std)

    def load_disturbance_models(self, output):

        if output is None:
            return

        self.disturb_estimators = []

        try:
            weights = torch.load('{}/gp_models.pkl'.format(output))
            train_x = torch.load('{}/gp_models_train_x.pkl'.format(output))
            train_y = torch.load('{}/gp_models_train_y.pkl'.format(output))
            for i in range(self.n_s):
                self.disturb_estimators.append(GPyDisturbanceEstimator(train_x, train_y[:, i], device=self.device))
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

