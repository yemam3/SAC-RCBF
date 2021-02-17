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
        self.get_h, self.get_dhdx = self.get_cbfs(env.hazards_locations, env.hazards_radius)
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

    def get_cbf_qp_constraints(self, u_nom, f_out, g_out, out, sigma_out):
        """Build up matrices required to solve qp
        Program specifically solves:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
                subject to G[u,eps]^T <= h

        Each control barrier certificate is of the form:
            dh/dx^T (f_out + g_out u) >= -gamma^b h_out^3 where out here is an output of the state.

        For example: in the case of unicycle dynamics, the CBF's are define with respect to the output
        p(state) defined as p(state) = [state[0] + l_p cos(state[2]), state[1] + l_p sin(state[2])]

        Parameters
        ----------
        u_nom : ndarray
            Nominal control input.
        f_out : ndarray
            f_out(s) at this state.
        g_out : ndarray
            g_out(s) at this state, dimensions (n_s, n_u)
        output : ndarray
            current output of state.
        sigma_out : ndarray
            standard deviation in additive disturbance after undergoing the output dynamics.

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


        # hs
        hs = self.get_h(out)
        dhdxs = self.get_dhdx(out)

        n_u = u_nom.shape[0]  # dimension of control inputs
        num_constraints = hs.shape[0] + 2 * n_u  # each cbf is a constraint, and we need to add actuator constraints (n_u of them)

        # Inequality constraints (G[u, eps] <= h)
        G = np.zeros((num_constraints, n_u + 1))  # the plus 1 is for epsilon (to make sure qp is always feasible)
        h = np.zeros(num_constraints)
        ineq_constraint_counter = 0

        # First let's add the cbf constraints
        for c in range(hs.shape[0]):

            # extract current affine cbf (h = h1^T x + h0)
            h_ = hs[c]
            dhdx_ = dhdxs[c]

            # Add inequality constraints
            G[ineq_constraint_counter, :n_u] = -dhdx_ @ g_out  # h1^Tg(x)
            G[ineq_constraint_counter, n_u] = -1  # for slack

            h[ineq_constraint_counter] = self.gamma_b * h_ + np.dot(dhdx_, f_out) + np.dot(dhdx_ @ g_out, u_nom) \
                                         - self.k_d * np.dot(np.abs(dhdx_), sigma_out)

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
        P = np.diag([1.e1, 1.e-4, 1e7])  # in the original code, they use 1e24 instead of 1e7, but quadprog can't handle that...
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
        """Returns CBF function h(x) and its derivative dh/dx(x) for each hazard. Note that the CBF is defined with
        with respect to an output of the state.

        Parameters
        ----------
        hazards_locations : list
            List of hazard-xy positions where each item is a list of length 2
        hazards_radius : float
            Radius of the hazards

        Returns
        -------
        get_h :
            List of cbfs each corresponding to a constraint. Each cbf is affine (h(s) = h1^Ts + h0), as such each cbf is represented
            as a tuple of two entries: h_cbf = (h1, h0) where h1 and h0 are ndarrays.
        get_dhdx :
        """

        hazards_locations = np.array(hazards_locations)
        collision_radius = hazards_radius + 0.07  # add a little buffer

        def get_h(state):
            return 0.5 * (np.sum((state - hazards_locations)**2, axis=1) - collision_radius**2)  # 1/2 * (||x - x_obs||^2 - r^2)

        def get_dhdx(state):
            dhdx = (state - hazards_locations)  # each row is dhdx_i for hazard i
            return dhdx

        return get_h, get_dhdx

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

    def get_min_h_val(self, state):
        """

        Parameters
        ----------
        state : ndarray
            Current State

        Returns
        -------
        min_h_val : float
            Minimum h(x) over all hs. If below 0, then at least 1 constraint is violated.

        """

        min_h_val = np.min(self.get_h(state))
        return min_h_val

