#!/usr/bin/env python3
"""
Easy startup script for the Minecraft PvP AI project
"""

import sys
import os


def main():
    print("=== Minecraft PvP AI Project ===")
    print()
    print("Choose what you want to run:")
    print("1. Game Demo with Bot - Interactive PvP game")
    print("2. Aiming Environment Demo - Text-based aiming training")
    print("3. Visual Aiming Demo - Visual aiming environment")
    print("4. Train Aiming AI - Train a new aiming model")
    print("5. Test Aiming AI - Test trained aiming model")
    print("6. Quit")

    while True:
        try:
            choice = input("\nEnter choice (1-6): ").strip()

            if choice == "1":
                print("\nStarting Game Demo...")
                print("Controls: WASD=move, Space=jump, Mouse=look, Click=attack")
                print("Press Q to quit, Esc to release mouse")
                os.system("python game_demo.py")

            elif choice == "2":
                print("\nStarting Aiming Environment Demo...")
                os.system("python run_aiming_demo.py")

            elif choice == "3":
                print("\nStarting Visual Aiming Demo...")
                print(
                    "Controls: 1/2/3=mode, Space=step, A=auto-toggle, R=reset, Q=quit"
                )
                model_path = input(
                    "Model path (press Enter for perfect aiming only): "
                ).strip()
                if model_path:
                    os.system(f"python visual_aiming_demo.py {model_path}")
                else:
                    os.system("python visual_aiming_demo.py")

            elif choice == "4":
                print("\nTraining Aiming AI...")
                print(
                    "This will take a while. The model will be saved as 'aiming_model_final.pth'"
                )
                from models.aiming.model import train_aiming_agent

                agent = train_aiming_agent(episodes=2000, render_every=100)
                agent.save("aiming_model_final.pth")
                print("Training completed! Model saved as 'aiming_model_final.pth'")

            elif choice == "5":
                model_path = input(
                    "Model path (default: aiming_model_final.pth): "
                ).strip()
                if not model_path:
                    model_path = "aiming_model_final.pth"
                print(f"\nTesting model: {model_path}")
                from models.aiming.model import test_aiming_agent

                try:
                    test_aiming_agent(model_path, episodes=5)
                except FileNotFoundError:
                    print(f"Model file {model_path} not found. Train a model first!")

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
            print(
                "Make sure all dependencies are installed: pip install -r requirements.txt"
            )


if __name__ == "__main__":
    main()
