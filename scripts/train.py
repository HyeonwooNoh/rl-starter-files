#!/usr/bin/env python3

import argparse
import gym
import gym_minigrid
import time
import datetime
import torch
import torch_ac
import sys

import utils
from model import ACModel
from model_aux import ACAuxModel
from model_aux_empower import ACAuxEmpowerModel

# Parse arguments

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo | ppo_aux | ppo_aux_empower (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--model_type", default="standard",
                    help="model type to use: standard | aux | aux_empower")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 10e7)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=0,
                    help="number of updates between two saves (default: 0, 0 means no saving)")
parser.add_argument("--tb", action="store_true", default=False,
                    help="log into Tensorboard")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=7e-4,
                    help="learning rate for optimizers (default: 7e-4)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--aux-loss-coef", type=float, default=0.5,
                    help="aux loss term coefficient (default: 0.5)")
parser.add_argument("--empower-value-loss-coef", type=float, default=0.5,
                    help="empower value loss term coefficient (default: 0.5)")
parser.add_argument("--empower-beta-coef", type=float, default=1.0,
                    help="empower beta coefficient (balance between entropy and action posterior) (default: 1.0)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer apha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of timesteps gradient is backpropagated (default: 1)\nIf > 1, a LSTM is added to the model to have memory")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
parser.add_argument("--prev_action", action="store_true", default=False,
                    help="add previous action as input to policy and value")
parser.add_argument("--aux_context", action="store_true", default=False,
                    help="add aux context as input to policy and value")
parser.add_argument("--aux_reward", action="store_true", default=False,
                    help="add aux reward")
parser.add_argument("--shaping_aux_reward", action="store_true", default=False,
                    help="use aux reward as a shaping function")
parser.add_argument("--manual_memory", action="store_true", default=False,
                    help="add manual memory size setting option")
parser.add_argument("--manual_memory_size", type=int, default=64,
                    help="semi memory size when is set manually")
parser.add_argument("--aux_reward_coef", type=float, default=0.1,
                    help="aux reward coef")
args = parser.parse_args()
args.mem = args.recurrence > 1

# Define run dir

suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = "{}_{}_seed{}_{}".format(args.env, args.algo, args.seed, suffix)
model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# Define logger, CSV writer and Tensorboard writer

logger = utils.get_logger(model_dir)
csv_file, csv_writer = utils.get_csv_writer(model_dir)
if args.tb:
    from tensorboardX import SummaryWriter
    tb_writer = SummaryWriter(model_dir)

# Log command and all script arguments

logger.info("{}\n".format(" ".join(sys.argv)))
logger.info("{}\n".format(args))

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environments

envs = []
for i in range(args.procs):
    env = gym.make(args.env)
    env.seed(args.seed + 10000*i)
    envs.append(env)

# Define obss preprocessor

obs_space, preprocess_obss = utils.get_obss_preprocessor(args.env, envs[0].observation_space, model_dir)

# Load training status

try:
    status = utils.load_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}

# Define actor-critic model

try:
    acmodel = utils.load_model(model_dir)
    logger.info("Model successfully loaded\n")
except OSError:
    if args.model_type == 'standard':
        acmodel = ACModel(obs_space, envs[0].action_space,
                          args.mem, args.text, args.prev_action,
                          args.manual_memory, args.manual_memory_size)
    elif args.model_type == 'aux':
        acmodel = ACAuxModel(obs_space, envs[0].action_space,
                             args.mem, args.text, args.prev_action,
                             args.manual_memory, args.manual_memory_size,
                             args.aux_context)
    elif args.model_type == 'aux_empower':
        acmodel = ACAuxEmpowerModel(obs_space, envs[0].action_space,
                                    args.mem, args.text, args.prev_action,
                                    args.manual_memory, args.manual_memory_size,
                                    args.aux_context)
    logger.info("Model successfully created\n")
logger.info("{}\n".format(acmodel))

if torch.cuda.is_available():
    acmodel.cuda()
logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

# Define actor-critic algo

if args.algo == "a2c":
    algo = torch_ac.A2CAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocess_obss)
elif args.algo == "ppo":
    algo = torch_ac.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
elif args.algo == "ppo_aux":
    algo = torch_ac.PPOAuxAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                               args.entropy_coef, args.value_loss_coef, args.aux_loss_coef, args.max_grad_norm, args.recurrence,
                               args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss,
                               use_aux_reward=args.aux_reward, aux_reward_coef=args.aux_reward_coef,
                               shaping_aux_reward=args.shaping_aux_reward)
elif args.algo == "ppo_aux_empower":
    algo = torch_ac.PPOAuxEmpowerAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                      args.entropy_coef, args.value_loss_coef, args.aux_loss_coef, args.max_grad_norm, args.recurrence,
                                      args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss,
                                      use_aux_reward=args.aux_reward, aux_reward_coef=args.aux_reward_coef,
                                      shaping_aux_reward=args.shaping_aux_reward,
                                      empower_beta_coef=args.empower_beta_coef,
                                      empower_value_loss_coef=args.empower_value_loss_coef)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

# Train model

num_frames = status["num_frames"]
total_start_time = time.time()
update = status["update"]

while num_frames < args.frames:
    # Update model parameters

    update_start_time = time.time()
    exps, logs1 = algo.collect_experiences()
    logs2 = algo.update_parameters(exps)
    logs = {**logs1, **logs2}
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs

    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - total_start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
            .format(*data))

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        if 'aux' in args.algo:
            header += ["aux_loss", "sample_entropy", "prev_aux_logprob"]
            data += [logs["aux_loss"], logs["sample_entropy"], logs["prev_aux_logprob"]]
        if 'aux_empower' in args.algo:
            header += ["empower_value", "empower_value_loss"]
            data += [logs["empower_value"], logs["empower_value_loss"]]

        if status["num_frames"] == 0:
            csv_writer.writerow(header)
        csv_writer.writerow(data)
        csv_file.flush()

        if args.tb:
            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        status = {"num_frames": num_frames, "update": update}

    # Save vocabulary, model and status

    if args.save_interval > 0 and update % args.save_interval == 0:
        if hasattr(preprocess_obss, 'vocab'):
            preprocess_obss.vocab.save()

        if torch.cuda.is_available():
            acmodel.cpu()
        utils.save_model(acmodel, model_dir)
        logger.info("Model successfully saved")
        if torch.cuda.is_available():
            acmodel.cuda()

        utils.save_status(status, model_dir)
