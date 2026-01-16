import torch
from torch import nn

import flappy_bird_gymnasium
import gymnasium

from dqn import DQN
from experience_replay import ReplayMemory

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib
import itertools
import argparse
import random
import yaml
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

matplotlib.use("Agg")

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

RUNS_DIR = 'runs'
MODELS_DIR = os.path.join(RUNS_DIR, "chkpt")
GRAPHS_DIR = os.path.join(RUNS_DIR, "graphs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

from config import HYPERPARAMETERS_PATH

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(self, hyperparameters_set, target=None):
        with open(HYPERPARAMETERS_PATH, "r") as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameters_set]

        self.env_id             = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size    = hyperparameters['mini_batch_size']
        self.epsilon_init       = hyperparameters['epsilon_init']
        self.epsilon_decay      = hyperparameters['epsilon_decay']
        self.epsilon_min        = hyperparameters['epsilon_min']
        self.sync_threshold     = hyperparameters['sync_threshold']
        self.stop_on            = hyperparameters['stop_on']
        self.discount_factor    = hyperparameters['discount_factor']
        self.learning_rate      = hyperparameters['learning_rate']

        self.TARGET_MODEL_FILE = target
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.enable_dueling_dqn = hyperparameters.get("enable_dueling_dqn", False)
        self.ddqn               = hyperparameters.get("enable_ddqn", False)
        
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict

        self.loss_fn   = nn.MSELoss()
        self.optimizer = None

        self.MODEL_FILE = os.path.join(MODELS_DIR, f'{hyperparameters_set}.pt')
        self.GRAPH_FILE = os.path.join(GRAPHS_DIR, f'{hyperparameters_set}.png')
        self.LOG_FILE   = os.path.join(GRAPHS_DIR, f'{hyperparameters_set}.log')

    def run(self, is_training, render):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gymnasium.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)

        num_states = env.observation_space.shape[0]
        num_action = env.action_space.n

        policy_dqn = DQN(num_states, num_action, self.fc1_nodes, self.enable_dueling_dqn).to(device)

        if is_training:
            epsilon = self.epsilon_init
            
            memory = ReplayMemory(self.replay_memory_size)

            #TODO Loading the target network
            target_dqn = DQN(num_states, num_action, self.fc1_nodes, self.enable_dueling_dqn).to(device)
            if self.TARGET_MODEL_FILE:
                target_dqn.load_state_dict(torch.load(self.TARGET_MODEL_FILE, map_location=device))
                
                log_message = datetime.now().strftime(DATE_FORMAT) + " Loaded target network."
                print(log_message)
                with open(self.LOG_FILE, "a") as f:
                    f.write(log_message)

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), self.learning_rate)

            rewards_per_episode = []
            epsilon_history = []

            step_count=0

            best_reward = -9999999

        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for epoch in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0
            pipes_count=0

            while (not terminated and episode_reward < self.stop_on):
                # Next action:
                # (feed the observation to your agent here)
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, truncated, info = env.step(action.item())
                
                episode_reward += reward
                if reward == 1.0:
                    pipes_count+=1

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated)) #Storing the experience in memory

                    step_count+=1


                state = new_state

            if is_training:
                rewards_per_episode.append(episode_reward)

                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {epoch}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=5):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    if step_count > self.sync_threshold:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0
            else:
                print(f"{datetime.now().strftime(DATE_FORMAT)} Episode reward: " + str(episode_reward))
                print(f"               Episode pipes passed: " + str(pipes_count))

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*batch)

        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        with torch.no_grad():
            #TODO// To make is Double-DQN -> needs to change max() to the best action of policy network
            if self.ddqn:
                best_actions = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1-terminations) * self.discount_factor * target_dqn(new_states).gather(dim=1, index=best_actions.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1-terminations) * self.discount_factor * target_dqn(new_states).max(dim=1)[0]

        policy_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        self.loss = self.loss_fn(target_q, policy_q)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing or training")
    parser.add_argument("hyperparameters", help="")
    parser.add_argument("--train", help="Training mode", action="store_true")

    #TODO// Creat a target network arg parser (optional)
    parser.add_argument("--target", type=str, default=None, help="Supervising target network models checkpoint path", nargs='?')
    args = parser.parse_args()

    agent = Agent(hyperparameters_set=args.hyperparameters, target=args.target)

    if args.train:
        agent.run(is_training=True, render=False)
    else:
        agent.run(is_training=False, render=True)
