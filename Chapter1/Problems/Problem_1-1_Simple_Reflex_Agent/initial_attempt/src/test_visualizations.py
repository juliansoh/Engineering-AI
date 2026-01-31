"""
Test Script for Agent Comparison Visualizations
===============================================

Simple test to verify histograms and summary visualizations work properly.
"""

import random
import sys
import os
import numpy as np
from typing import Dict

# Add the warehouse agent source directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../Warehouse_Agent/src'))

from warehouse_env import WarehouseEnv
from warehouse_agent_reflex import SimpleReflexAgent
sys.path.append('../../../Warehouse_Agent/src')
from warehouse_agent_greedy import GreedyManhattanAgent


def run_quick_test_episodes(agent, agent_name: str, num_episodes: int = 10) -> Dict:
    """Quick test of agent episodes for visualization testing"""
    print(f"🧪 Testing {agent_name} Agent ({num_episodes} episodes)...")
    
    all_steps = []
    all_batteries = []
    all_rewards = []
    successful_episodes = 0
    
    for episode in range(num_episodes):
        # Initialize fresh agent
        if agent_name == "Greedy":
            test_agent = GreedyManhattanAgent()
        else:
            test_agent = SimpleReflexAgent()
            
        # Create environment
        env = WarehouseEnv(max_steps=50)  # Short episodes
        obs = env.reset(randomize=True)
        
        # Random start
        start_pos = env._random_empty_cell()
        env.state.robot_pos = start_pos
        obs = env._observe()
        
        # Run episode
        step_count = 0
        total_reward = 0
        
        while step_count < 50:
            action = test_agent.act(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Collect stats
        all_steps.append(step_count)
        all_batteries.append(obs['battery'])
        all_rewards.append(total_reward)
        
        if terminated:
            successful_episodes += 1
    
    # Calculate statistics
    success_rate = successful_episodes / num_episodes * 100
    
    return {
        'agent_name': agent_name,
        'success_rate': success_rate,
        'successful_episodes': successful_episodes,
        'failed_episodes': num_episodes - successful_episodes,
        'all_steps': all_steps,
        'all_batteries': all_batteries,
        'all_rewards': all_rewards,
        'mean_steps_all': np.mean(all_steps),
        'mean_battery_all': np.mean(all_batteries),
        'mean_reward_all': np.mean(all_rewards)
    }


def test_summary_visualization(greedy_stats: Dict, reflex_stats: Dict) -> None:
    """Test the 3-subplot summary visualization"""
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError:
        print("❌ Matplotlib not available")
        return
    except Exception as e:
        print(f"❌ Matplotlib error: {e}")
        return
    
    print("📊 Creating Summary Visualization Test...")
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Agent Comparison Test - Summary', fontsize=14, fontweight='bold')
    
    # Colors
    colors = {'Greedy': '#2E86AB', 'Reflex': '#A23B72'}
    
    # 1. Success Rate Bar Chart
    agents = ['Greedy', 'Reflex']
    success_rates = [greedy_stats['success_rate'], reflex_stats['success_rate']]
    bars = ax1.bar(agents, success_rates, color=[colors['Greedy'], colors['Reflex']], alpha=0.7)
    
    for bar, value in zip(bars, success_rates):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom')
    
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate Comparison')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Episode Length Box Plot
    episode_data = [greedy_stats['all_steps'], reflex_stats['all_steps']]
    box_plot = ax2.boxplot(episode_data, labels=agents, patch_artist=True)
    box_plot['boxes'][0].set_facecolor(colors['Greedy'])
    box_plot['boxes'][1].set_facecolor(colors['Reflex'])
    
    ax2.set_ylabel('Episode Length (Steps)')
    ax2.set_title('Episode Length Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Battery Histograms
    ax3.hist(greedy_stats['all_batteries'], bins=8, alpha=0.6, 
            color=colors['Greedy'], label='Greedy', density=True)
    ax3.hist(reflex_stats['all_batteries'], bins=8, alpha=0.6, 
            color=colors['Reflex'], label='Reflex', density=True)
    
    ax3.set_xlabel('Final Battery Level')
    ax3.set_ylabel('Density')
    ax3.set_title('Final Battery Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    try:
        print("📈 Showing plots...")
        plt.show(block=True)  # Block to keep window open
        print("✅ Summary visualization test completed!")
    except Exception as e:
        print(f"❌ Display error: {e}")


def test_individual_histograms(agent_stats: Dict) -> None:
    """Test individual agent histograms"""
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        print("✅ Matplotlib ready for individual histograms")
    except Exception as e:
        print(f"❌ Matplotlib setup error: {e}")
        return
    
    agent_name = agent_stats['agent_name']
    print(f"📊 Creating histograms for {agent_name} Agent...")
    
    # Create 2x2 subplot for histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{agent_name} Agent - Histogram Analysis', fontsize=14, fontweight='bold')
    
    # 1. Episode Length Histogram
    ax = axes[0, 0]
    ax.hist(agent_stats['all_steps'], bins=8, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(agent_stats['mean_steps_all'], color='red', linestyle='--', 
              label=f'Mean: {agent_stats["mean_steps_all"]:.1f}')
    ax.set_xlabel('Episode Length (Steps)')
    ax.set_ylabel('Frequency')
    ax.set_title('Episode Length Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Battery Level Histogram
    ax = axes[0, 1]
    ax.hist(agent_stats['all_batteries'], bins=8, alpha=0.7, color='lightgreen', edgecolor='black')
    ax.axvline(agent_stats['mean_battery_all'], color='red', linestyle='--', 
              label=f'Mean: {agent_stats["mean_battery_all"]:.1f}')
    ax.set_xlabel('Final Battery Level')
    ax.set_ylabel('Frequency')
    ax.set_title('Final Battery Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Reward Histogram
    ax = axes[1, 0]
    ax.hist(agent_stats['all_rewards'], bins=8, alpha=0.7, color='plum', edgecolor='black')
    ax.axvline(agent_stats['mean_reward_all'], color='red', linestyle='--', 
              label=f'Mean: {agent_stats["mean_reward_all"]:.2f}')
    ax.set_xlabel('Total Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Success Rate Pie Chart
    ax = axes[1, 1]
    success_count = agent_stats['successful_episodes']
    failure_count = agent_stats['failed_episodes']
    labels = ['Success', 'Failure']
    sizes = [success_count, failure_count]
    colors_pie = ['#28a745', '#dc3545']
    
    if success_count > 0 or failure_count > 0:
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, 
                                         autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Success Rate: {agent_stats["success_rate"]:.1f}%')
    
    plt.tight_layout()
    
    try:
        print(f"📈 Showing {agent_name} histograms...")
        plt.show(block=True)  # Block to keep window open
        print(f"✅ {agent_name} histogram test completed!")
    except Exception as e:
        print(f"❌ Display error: {e}")


def main():
    """Main test function"""
    print("🧪 Testing Agent Comparison Visualizations")
    print("=" * 50)
    
    # Run quick tests
    print("\n1. Running quick agent tests...")
    greedy_stats = run_quick_test_episodes(GreedyManhattanAgent(), "Greedy", 10)
    reflex_stats = run_quick_test_episodes(SimpleReflexAgent(), "Reflex", 10)
    
    print(f"\n📊 Greedy Results: {greedy_stats['success_rate']:.1f}% success")
    print(f"📊 Reflex Results: {reflex_stats['success_rate']:.1f}% success")
    
    # Test individual histograms
    print("\n2. Testing individual histograms...")
    test_individual_histograms(greedy_stats)
    test_individual_histograms(reflex_stats)
    
    # Test summary visualization
    print("\n3. Testing summary visualization...")
    test_summary_visualization(greedy_stats, reflex_stats)
    
    print("\n✅ All visualization tests completed!")


if __name__ == "__main__":
    main()