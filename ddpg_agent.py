# DDPG for Pendulum-v1 control problem: agent 
# Born time: 2023-12-26
# Latest update: 2023-12-26
# Dylan

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
TAU = 0.005
MEMORY_SIZE = 100000
batch_size = 64


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Pendulum action space is [-2, 2]
        return x * 2


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Replay Buffer
class ReplayMemory:
    def __init__(self, capacity):  # 构造函数
        self.buffer = deque(maxlen=capacity)  # deque是双向队列，可以从两端append和pop

    def add_memo(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)  # np.expand_dims()是为了增加一个维度，从(3,)变成(1,3)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))  # 从右端插入
        # print(self.buffer)

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        # * is to unpack the list; zip is to combine the elements of the tuple
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
        '''
        对state使用 np.concatenate()函数是因为state是一个list，里面的元素是ndarray，所以要把它们拼接起来
        '''

    def __len__(self):  # a special method to get the length of the buffer
        return len(self.buffer)


# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)  # move nn to device
        self.actor_target = Actor(state_dim, action_dim).to(device)  # same structure as actor
        self.actor_target.load_state_dict(self.actor.state_dict())  # copy the current nn's weights of actor
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)  # retrieves the parameters

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayMemory(MEMORY_SIZE)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # unsqueeze(0) add a dimension from (3,) to (1,3)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]  # detach the tensor from the current graph and convert it to numpy
        '''
        .cpu() is a method that moves a tensor from GPU memory to CPU memory. 
        This is useful if you want to perform operations on the tensor using NumPy on the CPU.
        '''

    def update(self):
        if len(self.replay_buffer) < batch_size:
            return  # skip the update if the replay buffer is not filled enough

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(np.vstack(actions)).to(device)
        # actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Critic update
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states,
                                      next_actions.detach())  # .detach() means the gradient won't be backpropagated to the actor
        target_Q = rewards + (GAMMA * target_Q * (1 - dones))
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())  # nn.MSELoss() means Mean Squared Error
        self.critic_optimizer.zero_grad()  # .zero_grad() clears old gradients from the last step
        critic_loss.backward()  # .backward() computes the derivative of the loss
        self.critic_optimizer.step()  # .step() is to update the parameters

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()  # .mean() is to calculate the mean of the tensor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
