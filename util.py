import numpy as np
import os
import torch
from torch.autograd import Variable
import math

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def prRed(prt): print("\033[91m {}\033[00m".format(prt))


def prGreen(prt): print("\033[92m {}\033[00m".format(prt))


def prYellow(prt): print("\033[93m {}\033[00m".format(prt))


def prLightPurple(prt): print("\033[94m {}\033[00m".format(prt))


def prPurple(prt): print("\033[95m {}\033[00m".format(prt))


def prCyan(prt): print("\033[96m {}\033[00m".format(prt))


def prLightGray(prt): print("\033[97m {}\033[00m".format(prt))


def prBlack(prt): print("\033[98m {}\033[00m".format(prt))


def mat_to_euler_2d(rot_mat):
    """
    rot_mat has shape:
                [[c -s  0],
                 [s  c  0],
                 [0  0  1]]
    """

    theta = np.arcsin(rot_mat[1, 0])
    return theta


def euler_to_mat_2d(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([[c, -s], [s, c]])


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), requires_grad=requires_grad
    ).type(dtype)


def scale_action(action, action_lb, action_ub):
    act_k = (action_ub - action_lb) / 2.
    act_b = (action_ub + action_lb) / 2.
    if torch.is_tensor(action):
        return to_tensor(act_k) * action + to_tensor(act_b)
    return act_k * action + act_b


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


def get_wrapped_policy(agent, cbf_wrapper, dynamics_model, compensator=None, warmup=False, action_space=None,
                       policy_eval=False):
    # Get output functions p(x) dynamics
    get_f_out, get_g_out = dynamics_model.get_cbf_output_dynamics()  # get dynamics of output p(x) used by the CBF

    def wrapped_policy(observation):

        if warmup and action_space:
            action = action_space.sample()  # Sample random action
        else:
            action = agent.select_action(observation, evaluate=policy_eval)  # Sample action from policy

        if compensator:
            action_comp = compensator(observation)
        else:
            action_comp = 0
        state = dynamics_model.get_state(observation)
        # Get disturbance on output
        disturb_mean, disturb_std = dynamics_model.predict_disturbance(state)
        disturb_out_mean, disturb_out_std = dynamics_model.get_output_disturbance_dynamics(state,
                                                                                           disturb_mean, disturb_std)

        action_safe = cbf_wrapper.get_u_safe(action + action_comp, get_f_out(state) + disturb_out_mean,
                                             get_g_out(state),
                                             dynamics_model.get_output(state), disturb_out_std)
        # print('state = {}, action = {}, action_comp = {}, u_safe = {}'.format(state, action, action_comp, u_safe))
        return action + action_comp + action_safe

    return wrapped_policy
