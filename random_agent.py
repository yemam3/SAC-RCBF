import argparse
import safety_gym  # noqa
import numpy as np  # noqa
from build_env import build_env
from dynamics import DynamicsModel
from copy import deepcopy
import matplotlib.pyplot as plt
from util import *
from cbf import CBFLayer


def simple_controller(env, state, goal):
    goal_xy = goal[:2]
    goal_dist = -np.log(goal[2])  # the observation is np.exp(-goal_dist)
    v = 9e-3
    relative_theta = 0.7*np.arctan2(goal_xy[1], goal_xy[0])
    omega = relative_theta

    return np.clip(np.array([v, omega]), env.action_space.low, env.action_space.high)


def run_random(args):

    env = build_env(args)
    dynamics_model = DynamicsModel(env, args)
    get_f_out, get_g_out = dynamics_model.get_cbf_output_dynamics()  # get dynamics of output p(x) used by the CBF
    cbf_wrapper = CBFLayer(env, gamma_b=args.gamma_b, k_d=args.k_d)

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

    for i_step in range(3000):

        if done:
            print('Episode Return: %.3f \t Episode Cost: %.3f'%(ep_ret, ep_cost))
            ep_ret, ep_cost, ep_step = 0, 0, 0
            obs = env.reset()

        state = dynamics_model.get_state(obs)

        disturb_mean, disturb_std = dynamics_model.predict_disturbance(state)

        action = simple_controller(env, state, obs[[4, 5, 6]])  #TODO: observations 4,5 here indicated
        assert env.action_space.contains(action)
        out = dynamics_model.get_output(state)
        disturb_out_mean, disturb_out_std = dynamics_model.get_output_disturbance_dynamics(state, disturb_mean, disturb_std)
        action_safe = cbf_wrapper.get_u_safe(action, get_f_out(state) + disturb_out_mean, get_g_out(state), out, disturb_out_std)

        # Get confidence intervals
        next_state_pred, next_state_std = dynamics_model.predict_next_state(state, action + action_safe)
        next_state_lci_pred = next_state_pred - args.k_d * next_state_std
        next_state_uci_pred = next_state_pred + args.k_d * next_state_std

        # Env Step
        observation2, reward, done, info = env.step(action + action_safe)
        observation2 = deepcopy(observation2)

        # Update state and store transition for GP model learning
        next_state = dynamics_model.get_state(observation2)
        if ep_step % 2 == 0:
            dynamics_model.append_transition(state, action + action_safe, next_state)

        # test case focus here is on GPs
        for i in range(dynamics_model.n_s):
            model_actual_mean_history[i].append(next_state[i])
            model_prediction_mean_history[i].append(next_state_pred[i])
            model_prediction_lci_history[i].append(next_state_lci_pred[i])
            model_prediction_uci_history[i].append(next_state_uci_pred[i])
            model_prediction_std_history[i].append(next_state_std[i])

        # print('reward', reward)
        ep_ret += reward
        ep_cost += info.get('cost', 0)
        ep_step += 1
        # env.render()

        obs = observation2
        state = next_state

    # Initialize plot
    n_s = dynamics_model.n_s
    f, axs = plt.subplots(n_s, 1, figsize=(14, 8))
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
    plt.show()

    # Calculate avg_err for each state
    avg_err = []
    max_std = []
    for i in range(dynamics_model.n_s):
        mean_pred_ = np.array(model_prediction_mean_history[i])
        mean_act_ = np.array(model_actual_mean_history[i])
        avg_err.append(np.mean(np.abs(mean_pred_ - mean_act_)))
        max_std.append(np.max(model_prediction_std_history[i]))
    prGreen('Mean model error = {}'.format(avg_err))
    prGreen('Max std = {}'.format(max_std))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dynamics_mode', default='unicycle')
    parser.add_argument('--k_d', default=1.5, type=float)
    parser.add_argument('--gamma_b', default=100, type=float)
    parser.add_argument('--robot_xml', default='/xmls/unicycle_point.xml')
    parser.add_argument('--l_p', default=0.03, type=float, help="Point Robot only: Look-ahead distance for unicycle dynamics output.")
    args = parser.parse_args()

    args.robot_xml = os.getcwd() + args.robot_xml

    run_random(args)

