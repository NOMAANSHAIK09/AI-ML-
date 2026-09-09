import gymnasium as gym
import torch
import numpy as np
import random
from dqn import DQN
import matplotlib.pyplot as plt
from replay_memory import ReplayMemory
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.997

# -----------------------------
# Settings
# -----------------------------

batch_size = 64
memory_capacity = 10000
learning_rate = 0.00025
gamma = 0.99

num_episodes = 1000
target_update_frequency = 10

# -----------------------------
# Create environment
# -----------------------------

env = gym.make("CartPole-v1")


# -----------------------------
# Create neural network
# -----------------------------

policy_net = DQN()

target_net = DQN()

target_net.load_state_dict(policy_net.state_dict())

target_net.eval()

optimizer = torch.optim.Adam(
    policy_net.parameters(),
    lr=learning_rate
)


# -----------------------------
# Create replay memory
# -----------------------------

memory = ReplayMemory(memory_capacity)


# -----------------------------
# Loss function
# -----------------------------

loss_function = torch.nn.SmoothL1Loss()
rewards_history = []

# -----------------------------
# Training
# -----------------------------

for episode in range(num_episodes):

    # Start a new episode
    state, info = env.reset()

    total_reward = 0

    done = False

    while not done:

        # --------------------------------
        # Convert state to PyTorch tensor
        # --------------------------------

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )


        # --------------------------------
        # Get Q-values
        # --------------------------------

        q_values = policy_net(state_tensor)


        # --------------------------------
        # Choose action
        # --------------------------------

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(q_values).item()


        # --------------------------------
        # Perform action
        # --------------------------------

        next_state, reward, terminated, truncated, info = env.step(action)


        # --------------------------------
        # Check whether episode ended
        # --------------------------------

        done = terminated or truncated


        # --------------------------------
        # Store experience
        # --------------------------------

        experience = (
            state,
            action,
            reward,
            next_state,
            done
        )

        memory.push(experience)


        # --------------------------------
        # Move to next state
        # --------------------------------

        state = next_state

        total_reward += reward


        # --------------------------------
        # Wait until memory has enough data
        # --------------------------------

        if len(memory) < batch_size:
            continue


        # =================================
        # SAMPLE MINI-BATCH
        # =================================

        batch = memory.sample(batch_size)


        # --------------------------------
        # Separate experiences
        # --------------------------------

        states, actions, rewards, next_states, dones = zip(*batch)


        # --------------------------------
        # Convert to tensors
        # --------------------------------

        states = torch.tensor(
            np.array(states),
            dtype=torch.float32
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        )

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        )


        # =================================
        # CURRENT Q-VALUE
        # =================================

        q_values = policy_net(states)


        # Get Q-value of actions we actually took

        current_q = q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)


        # =================================
        # FUTURE Q-VALUE
        # =================================

        with torch.no_grad():

            next_q_values = target_net(next_states)

            future_q = next_q_values.max(
                dim=1
            )[0]


        # =================================
        # BELLMAN TARGET
        # =================================

        target_q = (
            rewards
            + gamma * future_q * (1 - dones)
        )


        # =================================
        # LOSS
        # =================================

        loss = loss_function(
            current_q,
            target_q
        )


        # =================================
        # BACKPROPAGATION
        # =================================

        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)

        optimizer.step()


    # --------------------------------
    # Episode information
    # --------------------------------
    epsilon = max(
    epsilon_min,
    epsilon * epsilon_decay
)
    # Update target network
    if (episode + 1) % target_update_frequency == 0:
        target_net.load_state_dict(
            policy_net.state_dict()
        )
        
    rewards_history.append(total_reward)
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(rewards_history[-10:])

        print(
            f"Episode: {episode + 1}/{num_episodes} "
            f"| Reward: {total_reward:.0f} "
            f"| Avg(10): {avg_reward:.1f} "
            f"| Epsilon: {epsilon:.3f}"
        )


torch.save(
    policy_net.state_dict(),
    "cartpole_dqn.pth"
)

print("Model saved successfully!")

plt.plot(rewards_history)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN CartPole Training")

plt.show()

# -----------------------------
# Close environment
# -----------------------------

env.close()