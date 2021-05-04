# import comet_ml at the top of your file
from comet_ml import Experiment

import argparse
import itertools
import torch

from rcbf_sac.sac_cbf import RCBF_SAC
from pytorch_sac.replay_memory import ReplayMemory
from dynamics import DynamicsModel
from build_env import *
import os

from util import prGreen, get_output_folder, prYellow
from evaluator import Evaluator
from rcbf_sac.generate_rollouts import generate_model_rollouts


def train(agent, env, dynamics_model, args, experiment=None):

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)
    memory_model = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_cost = 0
        episode_steps = 0
        done = False
        obs = env.reset()

        while not done:
            prYellow('Episode {} - step {} - eps_rew {} - eps_cost {}'.format(i_episode, episode_steps, episode_reward, episode_cost))
            state = dynamics_model.get_state(obs)

            # Generate Model rollouts
            if args.model_based and len(memory) > 0:
                memory_model = generate_model_rollouts(env, memory_model, memory, agent, dynamics_model, env.unwrapped.goal_pos[:2],
                                                       k_horizon=args.k_horizon,
                                                       batch_size=min(len(memory), args.rollout_batch_size),
                                                       warmup=args.start_steps > total_numsteps)

            # If using model-based RL then we only need to have enough data for the real portion of the replay buffer
            if len(memory) + len(memory_model) * args.model_based > args.batch_size:

                # Number of updates per step in environment
                for i in range(args.updates_per_step):

                    # Update parameters of all the networks
                    if args.model_based:
                        # Pick the ratio of data to be sampled from the real vs model buffers
                        real_ratio = max(min(args.real_ratio, len(memory) / args.batch_size), 1 - len(memory_model) / args.batch_size)
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                             args.batch_size,
                                                                                                             updates,
                                                                                                             dynamics_model,
                                                                                                             memory_model,
                                                                                                             real_ratio)
                    else:
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                           args.batch_size,
                                                                                                           updates,
                                                                                                           dynamics_model)

                    if experiment:
                        experiment.log_metric('loss/critic_1', critic_1_loss, updates)
                        experiment.log_metric('loss/critic_2', critic_2_loss, step=updates)
                        experiment.log_metric('loss/policy', policy_loss, step=updates)
                        experiment.log_metric('loss/entropy_loss', ent_loss, step=updates)
                        experiment.log_metric('entropy_temperature/alpha', alpha, step=updates)
                    updates += 1

            if args.start_steps > total_numsteps:
                action = agent.select_action(obs, dynamics_model, warmup=True)  # Sample action from policy
            else:
                action = agent.select_action(obs, dynamics_model)  # Sample action from policy

            next_obs, reward, done, info = env.step(action)  # Step
            if 'cost_exception' in info:
                prYellow('Cost exception occured.')
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            episode_cost += info.get('cost', 0)

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(obs, action, reward, next_obs, mask)  # Append transition to memory

            # Update state and store transition for GP model learning
            next_state = dynamics_model.get_state(next_obs)
            if episode_steps % 2 == 0:
                dynamics_model.append_transition(state, action, next_state)

            # [optional] save intermediate model
            if total_numsteps % int(args.num_steps / 10) == 0:
                agent.save_model(args.output)
                dynamics_model.save_disturbance_models(args.output)

            obs = next_obs

        if total_numsteps > args.num_steps:
            break

        if experiment:
            # Comet.ml logging
            experiment.log_metric('reward/train', episode_reward, step=i_episode)
            experiment.log_metric('cost/train', episode_cost, step=i_episode)
        prGreen("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, cost: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                             round(episode_reward, 2), round(episode_cost, 2)))

        # Evaluation
        if i_episode % 10 == 0 and args.eval is True:
            print('Size of replay buffers: real : {}, \t\t model : {}'.format(len(memory), len(memory_model)))
            avg_reward = 0.
            avg_cost = 0.
            episodes = 10
            for _ in range(episodes):
                obs = env.reset()
                episode_reward = 0
                episode_cost = 0
                done = False
                while not done:
                    action = agent.select_action(obs, dynamics_model, evaluate=True)
                    next_obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    episode_cost += info.get('cost', 0)
                    obs = next_obs

                avg_reward += episode_reward
                avg_cost += episode_cost
            avg_reward /= episodes
            avg_cost /= episodes
            if experiment:
                experiment.log_metric('avg_reward/test', avg_reward, step=i_episode)
                experiment.log_metric('avg_cost/test', avg_cost, step=i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}, Avg. Cost: {}".format(episodes, round(avg_reward, 2), round(avg_cost, 2)))
            print("----------------------------------------")


def test(num_episodes, agent, env, dynamics_model, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    dynamics_model.load_disturbance_models(model_path)

    def policy(observation):
        return agent.select_action(observation, dynamics_model, evaluate=True)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, dynamics_model=dynamics_model, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    # Environment Args
    parser.add_argument('--env-name', default="SafetyGym", help='Options are Unicycle or SafetyGym')
    parser.add_argument('--robot_xml', default='xmls/point.xml', help="SafetyGym Currently only supporting xmls/point.xml")
    parser.add_argument('--log_comet', action='store_true', dest='log_comet', help="Whether to log data")
    # SAC Args
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--visualize', action='store_true', dest='visualize', help='visualize env -only in available test mode')
    parser.add_argument('--output', default='output', type=str, help='')
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
                        help='Automatically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=12345, metavar='N',
                        help='random seed (default: 12345)')
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
    parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--validate_steps', default=1000, type=int, help='how many steps to perform a validate experiment')
    # CBF, Dynamics, Env Args
    parser.add_argument('--no_diff_qp', action='store_false', dest='diff_qp', help='Should the agent diff through the CBF?')
    parser.add_argument('--gp_model_size', default=2000, type=int, help='gp')
    parser.add_argument('--k_d', default=3.0, type=float)
    parser.add_argument('--gamma_b', default=100, type=float)
    parser.add_argument('--l_p', default=0.03, type=float,
                        help="Point Robot only: Look-ahead distance for unicycle dynamics output.")
    # Model Based Learning
    parser.add_argument('--model_based', action='store_true', dest='model_based', help='If selected, will use data from the model to train the RL agent.')
    parser.add_argument('--real_ratio', default=0.3, type=float, help='Portion of data obtained from real replay buffer for training.')
    parser.add_argument('--k_horizon', default=1, type=int, help='horizon of model-based rollouts')
    parser.add_argument('--rollout_batch_size', default=5, type=int, help='Size of initial states batch to rollout from.')
    args = parser.parse_args()

    args.output = get_output_folder(args.output, args.env_name)
    if args.resume == 'default':
        args.resume = os.getcwd() + '/output/{}-run0'.format(args.env_name)

    if args.mode == 'train' and args.log_comet:
        project_name = 'sac-rcbf-unicycle-environment' if args.env_name == 'Unicycle' else 'rl-cbf-unicycle-2'
        prYellow('Logging experiment on comet.ml!')
        # Create an experiment with your api key
        experiment = Experiment(
            api_key="FN3hKqygLp0oA32u1zSm7YtLF",
            project_name=project_name,
            workspace="yemam3",
        )
        # Log args on comet.ml
        experiment.log_parameters(vars(args))
        experiment_tags = ['MB' if args.model_based else 'MF',
                           str(args.batch_size) + '_batch',
                           str(args.updates_per_step) + '_step_updates',
                           'diff_qp' if args.diff_qp else 'reg_qp']
        print(experiment_tags)
        experiment.add_tags(experiment_tags)
    else:
        experiment = None

    # Environment
    env = build_env(args)

    # Agent
    agent = RCBF_SAC(env.observation_space.shape[0], env.action_space, env, args)
    dynamics_model = DynamicsModel(env, args)

    # Random Seed
    if args.seed > 0:
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        dynamics_model.seed(args.seed)

    evaluate = Evaluator(args.validate_episodes, args.validate_steps, args.output)

    # If model based, we warm up in the model too
    if args.model_based:
        args.start_steps /= (1 + args.rollout_batch_size)

    if args.mode == 'train':
        train(agent, env, dynamics_model, args, experiment)
    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, dynamics_model, evaluate, args.resume, visualize=False, debug=True)

    env.close()

