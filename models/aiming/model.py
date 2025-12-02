import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from enviroments.aiming.enviroment import AimingEnv


class AimingDQN(nn.Module):
    """
    Deep Q-Network for aiming task.
    Input: [distance to target, yaw difference, pitch difference, next yaw diff, next pitch diff]
    Output: Q-values for discrete actions (yaw/pitch adjustments)
    """

    def __init__(
        self, input_size=2, hidden_size=128, num_yaw_actions=11, num_pitch_actions=11
    ):
        super(AimingDQN, self).__init__()

        self.num_yaw_actions = num_yaw_actions
        self.num_pitch_actions = num_pitch_actions
        self.total_actions = num_yaw_actions * num_pitch_actions

        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, self.total_actions)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def get_action_from_index(self, action_index):
        """Convert action index to (dyaw, dpitch) values"""
        yaw_idx = action_index % self.num_yaw_actions
        pitch_idx = action_index // self.num_yaw_actions

        # Map indices to actual angle changes (in radians)
        max_delta = np.radians(5.0)  # 5 degrees max
        yaw_values = np.linspace(-max_delta, max_delta, self.num_yaw_actions)
        pitch_values = np.linspace(-max_delta, max_delta, self.num_pitch_actions)

        dyaw = yaw_values[yaw_idx]
        dpitch = pitch_values[pitch_idx]

        return np.array([dyaw, dpitch], dtype=np.float32)

    def get_index_from_action(self, action):
        """Convert (dyaw, dpitch) to action index (for training)"""
        dyaw, dpitch = action
        max_delta = np.radians(5.0)

        # Map continuous values to discrete indices
        yaw_values = np.linspace(-max_delta, max_delta, self.num_yaw_actions)
        pitch_values = np.linspace(-max_delta, max_delta, self.num_pitch_actions)

        yaw_idx = np.argmin(np.abs(yaw_values - dyaw))
        pitch_idx = np.argmin(np.abs(pitch_values - dpitch))

        return pitch_idx * self.num_yaw_actions + yaw_idx


class AimingAgent:
    """
    DQN Agent for aiming task with experience replay and target network
    """

    def __init__(
        self,
        state_size=2,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = AimingDQN(state_size).to(self.device)
        self.target_network = AimingDQN(state_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

        # Update target network
        self.update_target_network()
        self.target_update_freq = 100
        self.step_count = 0

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action_idx, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action_idx, reward, next_state, done))

    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            # Random action
            action_idx = np.random.randint(0, self.q_network.total_actions)
        else:
            # Greedy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()

        # Convert to continuous action
        action = self.q_network.get_action_from_index(action_idx)
        return action, action_idx

    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([e[4] for e in batch])).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def save(self, filepath):
        """Save model weights"""
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self.step_count,
            },
            filepath,
        )

    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.step_count = checkpoint["step_count"]


def train_aiming_agent(episodes=5000, render_every=100):
    """Train the aiming agent"""

    env = AimingEnv()
    agent = AimingAgent(epsilon_decay=0.8)

    scores = deque(maxlen=100)

    for episode in range(episodes):
        total_reward = 0
        for _ in range(100):
            state, _ = env.reset()
            steps = 0

            while True:
                action, action_idx = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.remember(state, action_idx, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1

                if done:
                    break

        scores.append(total_reward)

        # Train the agent
        loss = agent.replay()

        # Print progress
        if episode % render_every == 0:
            avg_score = np.mean(scores)
            print(
                f"Episode: {episode}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}"
            )
            if loss is not None:
                print(f"Loss: {loss:.4f}")

        # Save model periodically
        if episode % 25 == 0 and episode > 0:
            agent.save(f"aiming_model_episode_{episode}.pth")

    return agent


def test_aiming_agent(model_path, episodes=10):
    """Test a trained aiming agent"""
    from enviroments.aiming.enviroment import AimingEnv

    env = AimingEnv(render_mode="human")
    agent = AimingAgent()
    agent.load(model_path)

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        print(f"\n--- Episode {episode + 1} ---")

        while True:
            action, _ = agent.act(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            env.render()

            if terminated or truncated:
                break

        print(f"Episode {episode + 1} Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    # Train the agent
    print("Training aiming agent...")
    agent = train_aiming_agent(episodes=100, render_every=10)
    agent.save("aiming_model_final.pth")
    print("Training completed!")

    # Test the agent
    print("\nTesting trained agent...")
    test_aiming_agent("aiming_model_final.pth", episodes=5)
