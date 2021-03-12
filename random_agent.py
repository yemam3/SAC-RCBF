import argparse
import safety_gym  # noqa
import numpy as np  # noqa
from build_env import build_env
from dynamics import DynamicsModel
from copy import deepcopy
import matplotlib.pyplot as plt
from util import *
from cbf_cascade import CascadeCBFLayer


def simple_controller(env, state, goal):
    goal_xy = goal[:2]
    goal_dist = -np.log(goal[2])  # the observation is np.exp(-goal_dist)
    v = 0.02 * goal_dist
    relative_theta = 1.0*np.arctan2(goal_xy[1], goal_xy[0])
    omega = 1.0*relative_theta

    return np.clip(np.array([v, omega]), env.action_space.low, env.action_space.high)


def run_random(args):

    env = build_env(args)
    dynamics_model = DynamicsModel(env, args)
    cbf_wrapper = CascadeCBFLayer(env, gamma_b=args.gamma_b, k_d=args.k_d)

    obs = env.reset()
    done = False
    ep_ret = 0
    ep_cost = 0
    ep_step = 0

    # Data Saving purposes
    model_actual_mean_history = [[] for _ in range(dynamics_model.n_s)]
    model_prediction_mean_history = [[] for _ in range(dynamics_model.n_s)]
    model_prediction_lci_history = [[] for _ in range(dynamics_model.n_s)]
    model_prediction_uci_history = [[] for _ in range(dynamics_model.n_s)]
    model_prediction_std_history = [[] for _ in range(dynamics_model.n_s)]
    action_history = [[] for _ in range(dynamics_model.n_u)]

    for i_step in range(100000):

        if done:
            print('Episode Return: %.3f \t Episode Cost: %.3f'%(ep_ret, ep_cost))
            ep_ret, ep_cost, ep_step = 0, 0, 0
            obs = env.reset()

        state = dynamics_model.get_state(obs)

        disturb_mean, disturb_std = dynamics_model.predict_disturbance(state)

        action = simple_controller(env, state, obs[-3:])  #TODO: observations last 3 indicated
        # action = 2*np.random.rand(2) - 1.0
        assert env.action_space.contains(action)
        action_safe = cbf_wrapper.get_u_safe(action, state, disturb_mean, disturb_std)
        # action_safe = np.array([0.0, 0.0])

        # Get confidence intervals
        next_state_pred, next_state_std = dynamics_model.predict_next_state(state, action + action_safe)
        next_state_lci_pred = next_state_pred - args.k_d * next_state_std
        next_state_uci_pred = next_state_pred + args.k_d * next_state_std

        # Env Step
        observation2, reward, done, info = env.step(action + action_safe)
        observation2 = deepcopy(observation2)

        # Update state and store transition for GP model learning
        next_state = dynamics_model.get_state(observation2)
        # if ep_step % 2 == 0:
        #     dynamics_model.append_transition(state, action + action_safe, next_state)

        # test case focus here is on GPs
        for i in range(dynamics_model.n_s):
            model_actual_mean_history[i].append(next_state[i])
            model_prediction_mean_history[i].append(next_state_pred[i])
            model_prediction_lci_history[i].append(next_state_lci_pred[i])
            model_prediction_uci_history[i].append(next_state_uci_pred[i])
            model_prediction_std_history[i].append(next_state_std[i])
        for i in range(dynamics_model.n_u):
            action_history[i].append(action[i] + action_safe[i])

        # print('reward', reward)
        ep_ret += reward
        ep_cost += info.get('cost', 0)
        ep_step += 1
        # env.render()

        obs = observation2
        state = next_state

    # Initialize plot
    n_s = dynamics_model.n_s
    n_u = dynamics_model.n_u
    f, axs = plt.subplots(n_s + n_u, 1, figsize=(20, 12))
    for i in range(n_s):
        axs[i].plot(model_actual_mean_history[i], 'k*')
        # Plot predictive means as blue line
        axs[i].plot(model_prediction_mean_history[i], 'b')
        # Shade between the lower and upper confidence bounds
        axs[i].fill_between(range(len(model_actual_mean_history[i])), model_prediction_lci_history[i],
                        model_prediction_uci_history[i], alpha=0.5)
        #ax.set_ylim([-10.0, 10.0])
        axs[i].legend(['Real', 'Predicted', 'Confidence'])
        axs[i].set_ylabel('x_{}'.format(i))
        axs[i].set_xlabel('Step')

    # Plot action history
    for i in range(n_u):
        # Plot predictive means as blue line
        axs[n_s + i].plot(action_history[i], 'k')
        axs[n_s + i].set_ylabel('u_{}'.format(i))
        axs[n_s + i].set_xlabel('Step')
    plt.show()

    # def get_best_fit(X, y, guess=None):
    #
    #     if guess is None:
    #         guess = np.zeros(X.shape[1])
    #
    #     from scipy.optimize import minimize
    #
    #     def fit(X, params):
    #         return X.dot(params)
    #
    #     def cost_function(params, X, y):
    #         return np.sum(np.abs(y - fit(X, params)))
    #
    #     output = minimize(cost_function, guess, method='Nelder-Mead', args=(X, y))
    #     prCyan(output)

    # ### least squares ###
    # v = np.array(model_actual_mean_history[3])[1:]
    # omega = np.array(model_actual_mean_history[-1])[1:]
    # theta = np.array(model_actual_mean_history[2])[1:]
    # cos_th = np.cos(theta)
    # sin_th = np.sin(theta)
    # prCyan('omega_dot')
    # omega_dot = np.diff(model_actual_mean_history[4]) / 0.002
    # A = np.vstack([np.array(action_history[1])[1:], np.array(model_actual_mean_history[4])[:-1]]).T
    # get_best_fit(A, omega_dot)
    # prCyan('v_dot')
    # v_dot = np.diff(model_actual_mean_history[3]) / 0.002
    # A = np.vstack([np.array(action_history[0])[1:], np.array(model_actual_mean_history[3])[:-1]]).T
    # get_best_fit(A, v_dot)
    # prCyan('x_dot')
    # x_dot = np.diff(model_actual_mean_history[0]) / 0.002
    # A = np.vstack([v*cos_th]).T
    # get_best_fit(A, x_dot)
    # prCyan('y_dot')
    # y_dot = np.diff(model_actual_mean_history[1]) / 0.002
    # A = np.vstack([v*sin_th]).T
    # get_best_fit(A, y_dot)
    # prCyan('theta_dot')
    # theta_diff = np.diff(model_actual_mean_history[2])
    # theta_dot = np.arctan2(np.sin(theta_diff), np.cos(theta_diff)) / 0.002
    # A = np.vstack([omega]).T
    # get_best_fit(A, theta_dot)
    # ###### Get Modelling Error and plot it #################
    # x_dot = np.diff(model_actual_mean_history[0]) / 0.002 - 9*v*cos_th
    # y_dot = np.diff(model_actual_mean_history[1]) / 0.002 - 9*v*sin_th
    # theta_dot = np.arctan2(np.sin(theta_diff), np.cos(theta_diff)) / 0.002 - 5.5*omega
    # v_dot = np.diff(model_actual_mean_history[3]) / 0.002 - 40 * np.array(action_history[0])[1:] + 20 * np.array(model_actual_mean_history[3][:-1])
    # omega_dot = np.diff(model_actual_mean_history[4]) / 0.002 - 1520 * np.array(action_history[1])[1:] + 500 * np.array(model_actual_mean_history[4][:-1])
    # plt.figure(3)
    # plt.title('omega_dot error')
    # plt.plot(np.array(action_history[-1])[1:], omega_dot, 'k*')
    # plt.show()
    # plt.figure(4)
    # plt.title('v_dot error')
    # plt.plot(np.array(action_history[-1])[1:], v_dot, 'k*')
    # plt.show()
    # plt.figure(5)
    # plt.title('x_dot error')
    # plt.plot(v*cos_th, x_dot, 'k*')
    # plt.show()
    # plt.figure(6)
    # plt.title('y_dot error')
    # plt.plot(v*sin_th, y_dot, 'k*')
    # plt.show()
    # plt.figure(7)
    # plt.title('theta_dot error')
    # plt.plot(omega, theta_dot, 'k*')
    # plt.show()


    # Calculate avg_err for each state
    avg_err = []
    max_err = []
    med_err = []
    max_std = []
    for i in range(dynamics_model.n_s):
        mean_pred_ = np.array(model_prediction_mean_history[i])
        mean_act_ = np.array(model_actual_mean_history[i])
        avg_err.append(np.mean(np.abs(mean_pred_ - mean_act_)))
        med_err.append(np.mean(np.median(mean_pred_ - mean_act_)))
        max_err.append(np.max(np.abs(mean_pred_ - mean_act_)))
        max_std.append(np.max(model_prediction_std_history[i]))
    prGreen('Mean model error = {},\t Median model error = {},\t Max model error = {}'.format(avg_err, med_err, max_err))
    prGreen('Max std = {}'.format(max_std))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="CustomSafeExp-PointGoal-v0",
                        help='Doesn''t really matter, just for saving purposes')
    parser.add_argument('--k_d', default=3.0, type=float)
    parser.add_argument('--gamma_b', default=100, type=float)
    parser.add_argument('--robot_xml', default='/xmls/unicycle_point.xml')
    parser.add_argument('--l_p', default=0.03, type=float, help="Point Robot only: Look-ahead distance for unicycle dynamics output.")
    args = parser.parse_args()

    args.robot_xml = os.getcwd() + args.robot_xml

    run_random(args)

