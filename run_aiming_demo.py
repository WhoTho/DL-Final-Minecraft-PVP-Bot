# run_aiming_demo.py
"""
Easy startup script for the aiming environment with visual feedback
"""

import sys
import numpy as np
from enviroments.aiming.enviroment import AimingEnv
from models.aiming.model import AimingAgent, train_aiming_agent, test_aiming_agent


def demo_perfect_aiming():
    """Demo showing perfect aiming using the environment's get_perfect_action method"""
    print("Demo: Perfect Aiming")
    print(
        "This shows what perfect aiming looks like using the environment's built-in perfect action."
    )
    print("-" * 60)

    env = AimingEnv(render_mode="human")

    for episode in range(3):
        state, _ = env.reset()
        total_reward = 0

        print(f"\n=== Episode {episode + 1} ===")

        for step in range(50):  # Run for 50 steps
            # Get perfect action
            perfect_action = env.get_perfect_action()

            state, reward, terminated, truncated, _ = env.step(perfect_action)
            total_reward += reward

            if step % 10 == 0:  # Render every 10 steps
                env.render()

            if terminated or truncated:
                break

        print(f"Episode {episode + 1} Total Reward: {total_reward:.2f}")


def demo_random_aiming():
    """Demo showing random aiming"""
    print("\nDemo: Random Aiming")
    print("This shows how random actions perform.")
    print("-" * 60)

    env = AimingEnv(render_mode="human")

    for episode in range(3):
        state, _ = env.reset()
        total_reward = 0

        print(f"\n=== Episode {episode + 1} ===")

        for step in range(50):
            # Random action
            action = env.action_space.sample()

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if step % 10 == 0:
                env.render()

            if terminated or truncated:
                break

        print(f"Episode {episode + 1} Total Reward: {total_reward:.2f}")


def train_new_model():
    """Train a new aiming model"""
    print("\nTraining New Aiming Model")
    print("This will train a DQN agent to aim at moving targets.")
    print("-" * 60)

    agent = train_aiming_agent(episodes=1000, render_every=100)
    agent.save("aiming_model_demo.pth")
    print("\nModel saved as 'aiming_model_demo.pth'")
    return agent


def test_trained_model(model_path="aiming_model_demo.pth"):
    """Test a trained model"""
    print(f"\nTesting Trained Model: {model_path}")
    print("This shows how the trained agent performs.")
    print("-" * 60)

    try:
        test_aiming_agent(model_path, episodes=3)
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Train a model first!")


def interactive_aiming_test():
    """Interactive test where you can see step-by-step what happens"""
    print("\nInteractive Aiming Test")
    print("This lets you step through the environment manually.")
    print("-" * 60)

    env = AimingEnv(render_mode="human")

    state, _ = env.reset()
    env.render()

    print("Commands:")
    print("  p - take perfect action")
    print("  r - take random action")
    print("  n - reset environment")
    print("  q - quit")
    print("  h - help")

    total_reward = 0

    while True:
        cmd = input("\nCommand: ").strip().lower()

        if cmd == "q":
            break
        elif cmd == "h":
            print("Commands: p=perfect, r=random, n=reset, q=quit, h=help")
        elif cmd == "n":
            state, _ = env.reset()
            total_reward = 0
            print("Environment reset!")
            env.render()
        elif cmd == "p":
            action = env.get_perfect_action()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            print(
                f"Perfect action taken. Reward: {reward:.3f}, Total: {total_reward:.3f}"
            )
            env.render()

            if terminated or truncated:
                print("Episode finished!")
        elif cmd == "r":
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            print(
                f"Random action taken. Reward: {reward:.3f}, Total: {total_reward:.3f}"
            )
            env.render()

            if terminated or truncated:
                print("Episode finished!")
        else:
            print("Unknown command. Type 'h' for help.")


def main():
    """Main menu for the aiming demo"""
    print("=== Minecraft PvP Aiming Environment Demo ===")
    print()
    print("This demo shows an AI learning environment for aiming at moving targets.")
    print("The agent must learn to adjust yaw and pitch to track a moving target.")
    print()
    print("Choose a demo mode:")
    print("1. Perfect Aiming Demo - See what optimal aiming looks like")
    print("2. Random Aiming Demo - See how random actions perform")
    print("3. Interactive Test - Manual step-through")
    print("4. Train New Model - Train a DQN agent")
    print("5. Test Trained Model - Load and test a saved model")
    print("6. Quit")

    while True:
        try:
            choice = input("\nEnter choice (1-6): ").strip()

            if choice == "1":
                demo_perfect_aiming()
            elif choice == "2":
                demo_random_aiming()
            elif choice == "3":
                interactive_aiming_test()
            elif choice == "4":
                train_new_model()
            elif choice == "5":
                model_path = input(
                    "Model path (default: aiming_model_demo.pth): "
                ).strip()
                if not model_path:
                    model_path = "aiming_model_demo.pth"
                test_trained_model(model_path)
            elif choice == "6":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-6.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
