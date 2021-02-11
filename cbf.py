import numpy as np
import argparse
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from dynamics import DynamicsModel
from quadprog import solve_qp

class CBFLayer:

    def __init__(self, env, gamma_b=0.5, k_d=1.5):
        """Constructor of CBFLayer.

        Parameters
        ----------
        env : gym.env
            Gym environment.
        gamma_b : float, optional
            gamma of control barrier certificate.
        k_d : float, optional
            confidence parameter desired (2.0 corresponds to ~95% for example).
        """

        self.env = env
        self.Hs = self.get_cbfs(env.hazards_locations, env.hazards_radius)
        self.u_min, self.u_max = self.get_control_bounds()
        self.gamma_b = gamma_b
        self.k_d = k_d

    def get_u_safe(self, u_nom, f, g, s, sigma):
        """Given the current state of the system, this function computes the control input necessary to render the nominal
        control `u_nom` safe (i.e. u_safe + u_nom is safe).

        Parameters
        ----------
        u_nom : ndarray
            Nominal control input.
        f : ndarray
            f(s) at this state.
        g : ndarray
            g(s) at this state, dimensions (n_s, n_u)
        s : ndarray
            current state.
        sigma : ndarray
            standard deviation in additive disturbance.

        Returns
        -------
        u_safe : ndarray
            Safe control input to be added to `u_nom` as such `env.step(u_nom + u_safe)`
        """

        P, q, G, h = self.get_cbf_qp_constraints(u_nom, f, g, s, sigma)
        u_safe = self.solve_qp(P, q, G, h)

        return u_safe

    def get_cbf_qp_constraints(self, u_nom, f, g, s, sigma):
        """Build up matrices required to solve qp using cvxopt.solvers.qp
        Program specifically solves:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
                subject to G[u,eps]^T <= h
                           A[u,eps]^T <= b

        Each control barrier certificate is of the form:
            h_cbf(s_t+1) + (1 - gamma_b)h(s_t) >= 0.

        Parameters
        ----------
        u_nom : ndarray
            Nominal control input.
        f : ndarray
            f(s) at this state.
        g : ndarray
            g(s) at this state, dimensions (n_s, n_u)
        s : ndarray
            current state.
        sigma : ndarray
            standard deviation in additive disturbance.

        Returns
        -------
        P : ndarray
            Quadratic cost matrix in qp (minimize_{u,eps} 0.5 * u^T P u + q^T u)
        q : ndarray
            Linear cost vector in qp (minimize_{u,eps} 0.5 * u^T P u + q^T u)
        G : ndarray
            Inequality constraint matrix (G[u,eps] <= h) of size (num_constraints, n_u + 1)
        h : ndarray
            Inequality constraint vector (G[u,eps] <= h) of size (num_constraints,)
        """

        n_u = u_nom.shape[0]  # dimension of control inputs
        num_constraints = len(
            self.Hs) + 2 * n_u  # each cbf is a constraint, and we need to add actuator constraints (n_u of them)

        # Inequality constraints (G[u, eps] <= h)
        G = np.zeros((num_constraints, n_u + 1))  # the plus 1 is for epsilon (to make sure qp is always feasible)
        h = np.zeros(num_constraints)
        ineq_constraint_counter = 0

        # First let's add the cbf constraints
        for c in range(len(self.Hs)):

            # extract current affine cbf (h = h1^T x + h0)
            h1, h0 = self.Hs[c]
            # Add inequality constraints
            G[ineq_constraint_counter, :n_u] = -h1 @ g  # h1^Tg(x)
            G[ineq_constraint_counter, n_u] = -1  # for slack

            h[ineq_constraint_counter] = self.gamma_b * h0 + np.dot(h1, f) + np.dot(h1 @ g, u_nom) \
                                         - (1 - self.gamma_b) * np.dot(h1, s) - self.k_d * np.dot(np.abs(h1), sigma)

            ineq_constraint_counter += 1

        # Second let's add actuator constraints
        for c in range(n_u):

            # u_max >= u_nom + u ---> u <= u_max - u_nom
            if self.u_max is not None:
                G[ineq_constraint_counter, c] = 1
                h[ineq_constraint_counter] = self.u_max[c] - u_nom[c]
                ineq_constraint_counter += 1

            # u_min <= u_nom + u ---> -u <= u_min - u_nom
            if self.u_min is not None:
                G[ineq_constraint_counter, c] = -1
                h[ineq_constraint_counter] = -self.u_min[c] + u_nom[c]
                ineq_constraint_counter += 1

        # Let's also build the cost matrices, vectors to minimize control effort and penalize slack
        P = np.diag([1., 1., 1e7])  # in the original code, they use 1e24 instead of 1e7, but quadprog can't handle that...
        q = np.zeros(n_u + 1)

        return P, q, G, h

    def solve_qp(self, P, q, G, h):
        """Solves:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
                subject to G[u,eps]^T <= h

        Parameters
        ----------
        P : ndarray
            Quadratic cost matrix
        q : ndarray
            Linear cost vector
        G : ndarray
            Inequality Constraint Matrix
        h : ndarray
            Inequality Constraint Vector

        Returns
        -------
        u_safe : ndarray
            The solution of the qp without the last dimension (the slack).
        """

        try:
            sol = solve_qp(P, q, -G.T, -h)
            u_safe = sol[0][:-1]
        except ValueError as e:
            print(P, q, G, h)
            raise e

        # if np.abs(sol[0][-1]) > 1e-1:
        #     print('CBF indicates constraint violation might occur.')

        return u_safe

    def get_cbfs(self, hazards_locations, hazards_radius):
        """Returns affine discrete CBFs.

        Parameters
        ----------
        hazards_locations : list
            List of hazard-xy positions where each item is a list of length 2
        hazards_radius : float
            Radius of the hazards

        Returns
        -------
        Hs : list
            List of cbfs each corresponding to a constraint. Each cbf is affine (h(s) = h1^Ts + h0), as such each cbf is represented
            as a tuple of two entries: h_cbf = (h1, h0) where h1 and h0 are ndarrays.
        """


        # These are the constraints that actually keep the system safe with the real model parameters
        # Hs = [(np.array([1, 0.055]), 0.3),
        #       (np.array([1, -0.055]), 0.3),
        #       (np.array([-1, 0.055]), 0.3),
        #       (np.array([-1, -0.055]), 0.3)]

        # These are the constraints they used in the paper, which always let the pendulum fall
        Hs = [(np.array([1, 0.05, 0.00]), 1),
              #(np.array([1, -0.05]), 1),
              #(np.array([-1, 0.05]), 1),
              (np.array([-1, -0.05, 0.00]), 1)]

        return Hs

    def get_control_bounds(self):
        """

        Returns
        -------
        u_min : ndarray
            min control input.
        u_max : ndarray
            max control input.
        """

        u_min = self.env.unwrapped.action_space.low
        u_max = self.env.unwrapped.action_space.high

        return u_min, u_max

    def get_min_h_val(self, s):
        """

        Parameters
        ----------
        s : ndarray
            Current State

        Returns
        -------
        min_h_val : float
            Minimum h(x) over all hs. If below 0, then at least 1 constraint is violated.

        """

        min_h_val = float("inf")
        for h in self.Hs:
            h1, h0 = h
            min_h_val = min(min_h_val, np.dot(h1, s) + h0)

        return min_h_val


if __name__ == "__main__":

    # Pick the environment
    parser = argparse.ArgumentParser(description='Arguments for testing dynamics model.')
    parser.add_argument('--env_name', help='Name of the gym environment to use.', default='Pendulum-v0')
    parser.add_argument('--use_inaccurate_parameters', action='store_false')
    parser.add_argument('--k_d', default=2.0, type=float, help='CBF confidence parameter.')
    parser.add_argument('--gamma_b', default=0.5, type=float, help='CBF conservativeness parameter.')
    args = vars(parser.parse_args())

    args['m'] = 1.4 if args['use_inaccurate_parameters'] else 1.0  # Mass of pendulum in Pendulum-v0 (1.0 default)
    args['l'] = 1.4 if args['use_inaccurate_parameters'] else 1.0  # Length of pendulum in Pendulum-v0 (1.0 default)

    print('Running simple code to check if our CBFs for the {} env actually keep us safe!'.format(args['env_name']))

    # Env and Dynamics Model go hand in hand, always
    env = tweak_env_parameters(gym.make(args['env_name']))
    dynamics_model = DynamicsModel(env, **args)  # our model of the dynamics
    get_f, get_g = dynamics_model.get_dynamics()  # get dynamics of discrete system x' = f(x) + g(x)u
    cbf_wrapper = CBFLayer(env, gamma_b=args['gamma_b'], k_d=args['k_d'])

    # State/Input/h_min history
    model_actual_mean_history = []
    model_prediction_lci_history = []
    model_prediction_uci_history = []
    h_min_history = []
    u_history = []
    num_unsafe_episodes = 0
    num_episodes = 10
    inaccurate_ci_count = 0  # how many steps is the actual next state outside our predicted CI

    for i_episode in tqdm(range(num_episodes), desc='Episode number'):

        # start from a specific initial state
        _ = env.reset()
        env.env.state = (np.random.random(2) - 0.5) * np.array([np.pi / 6, 0.01])
        env.state = env.env.state
        state = env.env.state
        obs = dynamics_model.get_obs(state)

        # For data logging purposes
        u_history.append([])
        h_min_history.append([])
        model_actual_mean_history.append([[] for _ in range(dynamics_model.n_s)])
        model_prediction_lci_history.append([[] for _ in range(dynamics_model.n_s)])
        model_prediction_uci_history.append([[] for _ in range(dynamics_model.n_s)])

        for t in range(200):
            # print('\tstep = {}, state = {}'.format(t, state))

            # Get nominal and safe actions
            u_nom = env.unwrapped.action_space.sample() * 1/2
            disturb_mean, disturb_std = dynamics_model.predict_disturbance(state)
            if not args['use_inaccurate_parameters']:
                disturb_std *= 0
            u_safe = cbf_wrapper.get_u_safe(u_nom, get_f(state) + disturb_mean, get_g(state), state, disturb_std)

            # Check if we caused a safety violation
            if cbf_wrapper.get_min_h_val(state) < -1e-3:
                print('\nSafety Violation, h_min = {}'.format(h_min_history[i_episode][-1]))
                num_unsafe_episodes += 1
                break

            # Step env and get state
            obs, reward, done, info = env.step(u_nom + u_safe)
            next_state = dynamics_model.get_state(obs)
            dynamics_model.append_transition(state, u_nom + u_safe, next_state)

            # Check if that next_state matched our predictions
            next_state_prior = get_f(state) + get_g(state) @ (u_nom + u_safe)
            next_state_lci_pred = next_state_prior + disturb_mean - args['k_d']*disturb_std
            next_state_uci_pred = next_state_prior + disturb_mean + args['k_d']*disturb_std
            if np.any(np.bitwise_or(next_state < next_state_lci_pred-1e-3, next_state > next_state_uci_pred+1e-3)):
                # print('\nActual next_state = {}, Problem occured at dims = {}'.format(next_state, ((np.bitwise_or(next_state < next_state_lci_pred, next_state > next_state_uci_pred)))))
                # print('Predicted L-CI = {}, U-CI = {}, mean = {}, std = {}'.format(next_state_lci_pred, next_state_uci_pred, next_state_prior + disturb_mean, disturb_std))
                # raise Exception('Prediction Error occurred. Actual next state was not in our prediction CI.')
                inaccurate_ci_count += 1

            # Save some data to plot later
            u_history[i_episode].append(u_nom + u_safe)
            h_min_history[i_episode].append(cbf_wrapper.get_min_h_val(next_state))
            for i in range(dynamics_model.n_s):
                model_actual_mean_history[i_episode][i].append(next_state[i])
                model_prediction_lci_history[i_episode][i].append(next_state_lci_pred[i])
                model_prediction_uci_history[i_episode][i].append(next_state_uci_pred[i])

            # Update current state
            state = next_state

            if done:
                # print("Episode finished after {} timesteps".format(t + 1))
                break

    # Plot All States against time
    for i_state in range(dynamics_model.n_s):
        for i_episode in range(len(model_actual_mean_history)):
            plt.plot(model_actual_mean_history[i_episode][i_state])
            plt.fill_between(range(len(model_prediction_lci_history[i_episode][i_state])), model_prediction_lci_history[i_episode][i_state], model_prediction_uci_history[i_episode][i_state], alpha=0.5)
        plt.xlabel('Step')
        plt.ylabel('State[{}]'.format(i_state))
        plt.title('State[{}] vs time'.format(i_state))
        plt.show()

    # Plot h against time
    for i in range(len(h_min_history)):
        plt.plot(h_min_history[i])
    plt.xlabel('Step')
    plt.ylabel('min h(x[t])')
    plt.title('Min h vs time')
    plt.show()

    # Plot u against time
    for i in range(len(u_history)):
        plt.plot(u_history[i])
    plt.xlabel('Step')
    plt.ylabel('u')
    plt.title('u vs time')
    plt.show()

    print('Number of safe episodes: {}/{} -> {}%'.format(num_episodes - num_unsafe_episodes, num_episodes,
                                                         100 * (1 - num_unsafe_episodes / num_episodes)))
    total_num_steps = sum([len(h_min_history[i]) for i in range(len(h_min_history))])
    print('Number of accurate-ci-steps: {}/{} -> {}%'.format(total_num_steps - inaccurate_ci_count, total_num_steps,
                                                         100 * (1 - inaccurate_ci_count / total_num_steps)))
    print('The inaccuracies mainly occur because we assume the disturbance is only on f(x), which doesn''t scale with u.')