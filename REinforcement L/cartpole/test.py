import gymnasium as gym
import torch

from dqn import DQN


# -----------------------------
# Settings
# -----------------------------

num_test_episodes = 5


# -----------------------------
# Create environment
# -----------------------------

env = gym.make(
    "CartPole-v1",
    render_mode="human"
)


# -----------------------------
# Create model
# -----------------------------

model = DQN()


# -----------------------------
# Load trained model
# -----------------------------

model.load_state_dict(
    torch.load(
        "cartpole_dqn.pth",
        weights_only=True
    )
)

model.eval()


# -----------------------------
# Test model
# -----------------------------

test_rewards = []


for episode in range(num_test_episodes):

    state, info = env.reset()

    done = False
    total_reward = 0

    while not done:

        # Convert state to tensor
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )

        # Get Q-values
        with torch.no_grad():
            q_values = model(state_tensor)

        # Always choose best action
        action = torch.argmax(q_values).item()

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Check if episode ended
        done = terminated or truncated

        # Move to next state
        state = next_state

        total_reward += reward

    test_rewards.append(total_reward)

    print(
        f"Test Episode: {episode + 1} "
        f"| Reward: {total_reward:.0f}"
    )


# -----------------------------
# Calculate average
# -----------------------------

average_reward = sum(test_rewards) / len(test_rewards)

print("-----------------------------")
print(f"Average Reward: {average_reward:.2f}")
print("-----------------------------")


env.close()