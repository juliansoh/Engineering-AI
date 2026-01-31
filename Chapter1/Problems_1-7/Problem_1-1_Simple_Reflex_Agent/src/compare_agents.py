"""
Agent Comparison Script
=====================

This script runs 50 episodes of both the Greedy Manhattan Agent and Simple Reflex Agent,
collecting performance metrics and creating comparative visualizations.

Author: Created for EAI Chapter 1 - Problem 1-1 Agent Comparison
"""

import sys
import os
import json
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

# Add the warehouse agent source directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../Warehouse_Agent/src'))

from warehouse_env import WarehouseEnv
from warehouse_agent_greedy import GreedyManhattanAgent
from warehouse_agent_reflex import SimpleReflexAgent


def run_agent_episodes(agent_class, agent_name: str, num_episodes: int = 50, max_steps: int = 100) -> Dict:
    """
    Run multiple episodes for a given agent and collect performance metrics.
    
    Args:
        agent_class: The agent class to instantiate
        agent_name: Name of the agent for logging
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary containing performance metrics for all episodes
    """
    print(f"\n🤖 Running {num_episodes} episodes for {agent_name}")
    print("=" * 60)
    
    # Initialize results dictionary
    results = {
        "agent_name": agent_name,
        "success": [],           # True/False for each episode
        "episode_length": [],    # Steps taken for each episode
        "final_battery": [],     # Final battery level for each episode
        "total_reward": [],      # Total reward for each episode
        "completed_episodes": 0,  # Count of successfully completed episodes
        "truncated_episodes": 0  # Count of episodes that hit time limit
    }
    
    for episode in range(num_episodes):
        print(f"  Episode {episode + 1}/{num_episodes}", end=" ")
        
        # Initialize environment and agent
        env = WarehouseEnv(max_steps=max_steps)
        agent = agent_class()
        
        # Reset environment with randomized layout and random start
        obs = env.reset(randomize=True)
        
        # Move robot to random starting position
        random_pos = env._random_empty_cell()
        env.state.robot_pos = random_pos
        obs = env._observe()
        
        # Run episode
        step_count = 0
        total_reward = 0
        
        while step_count < max_steps:
            # Get action from agent
            action = agent.act(obs, env)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            total_reward += reward
            
            # Check termination conditions
            if terminated:
                print("✅ SUCCESS")
                results["success"].append(True)
                results["completed_episodes"] += 1
                break
            elif truncated:
                print("❌ TIMEOUT")
                results["success"].append(False)
                results["truncated_episodes"] += 1
                break
        else:
            # This shouldn't happen with proper truncation, but just in case
            print("❌ MAX STEPS")
            results["success"].append(False)
            results["truncated_episodes"] += 1
        
        # Record episode metrics
        results["episode_length"].append(step_count)
        results["final_battery"].append(obs["battery"])
        results["total_reward"].append(total_reward)
    
    # Calculate summary statistics
    success_rate = (results["completed_episodes"] / num_episodes) * 100
    avg_steps = np.mean(results["episode_length"])
    avg_battery = np.mean(results["final_battery"])
    avg_reward = np.mean(results["total_reward"])
    
    print(f"\n📊 {agent_name} Summary:")
    print(f"   Success Rate: {success_rate:.1f}% ({results['completed_episodes']}/{num_episodes})")
    print(f"   Average Steps: {avg_steps:.1f}")
    print(f"   Average Final Battery: {avg_battery:.1f}")
    print(f"   Average Total Reward: {avg_reward:.2f}")
    
    return results


def visualize_agent_comparison(greedy_results: Dict, reflex_results: Dict, save_path: str = None):
    """
    Create visualization comparing the two agents with 3 subplots:
    1. Bar chart of success rates
    2. Box plots of episode lengths  
    3. Histograms of final battery levels
    
    Args:
        greedy_results: Results dictionary from greedy agent
        reflex_results: Results dictionary from reflex agent
        save_path: Optional path to save the figure
    """
    print(f"\n📈 Creating comparison visualizations...")
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Agent Performance Comparison: Greedy vs Reflex', fontsize=16, fontweight='bold')
    
    # Calculate success rates
    greedy_success_rate = (greedy_results["completed_episodes"] / len(greedy_results["success"])) * 100
    reflex_success_rate = (reflex_results["completed_episodes"] / len(reflex_results["success"])) * 100
    
    # Subplot 1: Bar chart of success rates
    ax1 = axes[0]
    agents = ['Greedy\nManhattan', 'Simple\nReflex']
    success_rates = [greedy_success_rate, reflex_success_rate]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax1.bar(agents, success_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rates', fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add grid
    ax1.grid(axis='y', alpha=0.3)
    
    # Subplot 2: Box plots of episode lengths
    ax2 = axes[1]
    episode_data = [greedy_results["episode_length"], reflex_results["episode_length"]]
    
    box_plot = ax2.boxplot(episode_data, labels=agents, patch_artist=True)
    
    # Color the box plots
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Episode Length (Steps)')
    ax2.set_title('Episode Length Distribution', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Subplot 3: Histograms of final battery levels
    ax3 = axes[2]
    
    # Create histograms
    ax3.hist(greedy_results["final_battery"], bins=15, alpha=0.6, color=colors[0], 
             label='Greedy Manhattan', edgecolor='black', linewidth=0.5)
    ax3.hist(reflex_results["final_battery"], bins=15, alpha=0.6, color=colors[1], 
             label='Simple Reflex', edgecolor='black', linewidth=0.5)
    
    ax3.set_xlabel('Final Battery Level')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Final Battery Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved visualization to: {save_path}")
    
    # Display the plot
    plt.show()
    
    # Print detailed comparison statistics
    print(f"\n📊 Detailed Comparison Statistics:")
    print(f"{'Metric':<25} {'Greedy':<15} {'Reflex':<15} {'Winner':<10}")
    print("-" * 70)
    
    # Success rate comparison
    greedy_wins = greedy_success_rate > reflex_success_rate
    print(f"{'Success Rate (%)':<25} {greedy_success_rate:<15.1f} {reflex_success_rate:<15.1f} {'Greedy' if greedy_wins else 'Reflex'}")
    
    # Episode length comparison (lower is better)
    greedy_avg_steps = np.mean(greedy_results["episode_length"])
    reflex_avg_steps = np.mean(reflex_results["episode_length"])
    steps_winner = "Greedy" if greedy_avg_steps < reflex_avg_steps else "Reflex"
    print(f"{'Avg Episode Length':<25} {greedy_avg_steps:<15.1f} {reflex_avg_steps:<15.1f} {steps_winner}")
    
    # Battery efficiency comparison (higher is better)
    greedy_avg_battery = np.mean(greedy_results["final_battery"])
    reflex_avg_battery = np.mean(reflex_results["final_battery"])
    battery_winner = "Greedy" if greedy_avg_battery > reflex_avg_battery else "Reflex"
    print(f"{'Avg Final Battery':<25} {greedy_avg_battery:<15.1f} {reflex_avg_battery:<15.1f} {battery_winner}")
    
    # Total reward comparison (higher is better)
    greedy_avg_reward = np.mean(greedy_results["total_reward"])
    reflex_avg_reward = np.mean(reflex_results["total_reward"])
    reward_winner = "Greedy" if greedy_avg_reward > reflex_avg_reward else "Reflex"
    print(f"{'Avg Total Reward':<25} {greedy_avg_reward:<15.2f} {reflex_avg_reward:<15.2f} {reward_winner}")


def save_results_to_json(greedy_results: Dict, reflex_results: Dict, filename: str = "agent_comparison_results.json"):
    """
    Save the comparison results to a JSON file for later analysis.
    
    Args:
        greedy_results: Results dictionary from greedy agent
        reflex_results: Results dictionary from reflex agent
        filename: Name of the JSON file to save
    """
    comparison_data = {
        "experiment_info": {
            "num_episodes_per_agent": len(greedy_results["success"]),
            "max_steps_per_episode": 100
        },
        "greedy_agent": greedy_results,
        "reflex_agent": reflex_results
    }
    
    with open(filename, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"📁 Results saved to: {filename}")


def main():
    """
    Main function to run the agent comparison experiment.
    """
    print("🔬 Agent Comparison Experiment")
    print("=" * 50)
    print("Comparing Greedy Manhattan Agent vs Simple Reflex Agent")
    print("Running 50 episodes each with randomized environments")
    
    # Set random seed for reproducibility (optional)
    random.seed(42)
    np.random.seed(42)
    
    # Run episodes for both agents
    print("\n🚀 Starting comparison experiment...")
    
    # Run Greedy Manhattan Agent
    greedy_results = run_agent_episodes(
        agent_class=GreedyManhattanAgent,
        agent_name="Greedy Manhattan Agent",
        num_episodes=50,
        max_steps=100
    )
    
    # Run Simple Reflex Agent
    reflex_results = run_agent_episodes(
        agent_class=SimpleReflexAgent,
        agent_name="Simple Reflex Agent", 
        num_episodes=50,
        max_steps=100
    )
    
    # Create visualizations
    visualize_agent_comparison(greedy_results, reflex_results, save_path="agent_comparison.png")
    
    # Save results to JSON
    save_results_to_json(greedy_results, reflex_results)
    
    print(f"\n🎯 Experiment Complete!")
    print(f"   Total episodes run: {len(greedy_results['success']) + len(reflex_results['success'])}")
    print(f"   Results saved and visualized successfully")


if __name__ == "__main__":
    main()