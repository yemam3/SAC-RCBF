import argparse
import datetime
import itertools
import torch
from pytorch_sac.sac import SAC
from torch.utils.tensorboard import SummaryWriter
from pytorch_sac.replay_memory import ReplayMemory
from cbf import CBFLayer
from dynamics import DynamicsModel
from build_env import *
import os
from util import prGreen
from pathlib import Path

def train(agent, cbf_wrapper, env, dynamics_model, args):

    # Tensorboard
    writer = SummaryWriter(
        'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                      args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0

    # Get output functions p(x) dynamics
    get_f_out, get_g_out = dynamics_model.get_cbf_output_dynamics()  # get dynamics of output p(x) used by the CBF

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_cost = 0
        episode_steps = 0
        done = False
        obs = env.reset()

        while not done:

            state = dynamics_model.get_state(obs)
            disturb_mean, disturb_std = dynamics_model.predict_disturbance(state)
            out = dynamics_model.get_output(state)
            disturb_out_mean, disturb_out_std = dynamics_model.get_output_disturbance_dynamics(state, disturb_mean,
                                                                                               disturb_std)

            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(obs)  # Sample action from policy

            action_safe = cbf_wrapper.get_u_safe(action, get_f_out(state) + disturb_out_mean, get_g_out(state), out, disturb_out_std)

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                         args.batch_size,
                                                                                                         updates)
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temperature/alpha', alpha, updates)
                    updates += 1

            next_obs, reward, done, info = env.step(action + action_safe)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            episode_cost += info.get('cost', 0)

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(obs, action + action_safe, reward, next_obs, mask)  # Append transition to memory

            # Update state and store transition for GP model learning
            next_state = dynamics_model.get_state(next_obs)
            if episode_steps % 2 == 0:
                dynamics_model.append_transition(state, action + action_safe, next_state)

            obs = next_obs

        if total_numsteps > args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        writer.add_scalar('cost/train', episode_cost, i_episode)
        prGreen("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, cost: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                      round(episode_reward, 2), round(episode_cost, 2)))

        if i_episode % 10 == 0 and args.eval is True:
            avg_reward = 0.
            avg_cost = 0.
            episodes = 10
            for _ in range(episodes):
                obs = env.reset()
                episode_reward = 0
                episode_cost = 0
                done = False
                while not done:
                    state = dynamics_model.get_state(obs)
                    disturb_mean, disturb_std = dynamics_model.predict_disturbance(state)
                    out = dynamics_model.get_output(state)
                    disturb_out_mean, disturb_out_std = dynamics_model.get_output_disturbance_dynamics(state,
                                                                                                       disturb_mean, disturb_std)
                    action = agent.select_action(obs, evaluate=True)
                    action_safe = cbf_wrapper.get_u_safe(action, get_f_out(state) + disturb_out_mean, get_g_out(state),
                                                         out, disturb_out_std)
                    next_obs, reward, done, info = env.step(action + action_safe)
                    episode_reward += reward
                    episode_cost += info.get('cost', 0)

                    obs = next_obs
                avg_reward += episode_reward
                avg_cost += episode_cost
            avg_reward /= episodes
            avg_cost /= episodes
            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}, Avg. Cost: {}".format(episodes, round(avg_reward, 2), round(avg_cost, 2)))
            print("----------------------------------------")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="CustomSafeExp-PointGoal-v0",
                        help='Doesn''t really matter, just for saving purposes')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    # CBF, Dynamics, Env Args
    parser.add_argument('--dynamics_mode', default='unicycle')
    parser.add_argument('--k_d', default=1.5, type=float)
    parser.add_argument('--gamma_b', default=100, type=float)
    parser.add_argument('--robot_xml', default='/xmls/unicycle_point.xml')
    parser.add_argument('--l_p', default=0.03, type=float,
                        help="Point Robot only: Look-ahead distance for unicycle dynamics output.")
    args = parser.parse_args()

    args.robot_xml = str(Path(os.getcwd()).parent) + args.robot_xml

    # Environment
    env = build_env(args)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    cbf_wrapper = CBFLayer(env, gamma_b=args.gamma_b, k_d=args.k_d)
    dynamics_model = DynamicsModel(env, args)
    dynamics_model.seed(args.seed)

    # Train
    train(agent, cbf_wrapper, env, dynamics_model, args)

    env.close()

