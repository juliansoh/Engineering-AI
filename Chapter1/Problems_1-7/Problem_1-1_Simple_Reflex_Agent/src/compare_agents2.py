"""
Agent Comparison: Greedy vs Reflex
==================================

This script compares the performance of the Greedy Manhattan Agent and Simple Reflex Agent
using identical warehouse setups and comprehensive analysis with visualizations.

Features:
- Side-by-side visualization of both agents
- Identical environment conditions for fair comparison
- Statistical analysis with multiple visualization types
- Detailed logging and performance metrics

Author: Created for EAI Chapter 1 - Agent Comparison
"""

import random
import sys
import os
from typing import List, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass, asdict
import json
from pathlib import Path

# Add the warehouse agent source directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../Warehouse_Agent/src'))

from warehouse_env import WarehouseEnv
from warehouse_viz import replay_animation, _grid_to_rgb

# Import both agents
from warehouse_agent_reflex import SimpleReflexAgent
sys.path.append('../../../Warehouse_Agent/src')
from warehouse_agent_greedy import GreedyManhattanAgent


@dataclass
class EpisodeResult:
    """Data structure for storing episode results"""
    agent_name: str
    episode_id: int
    success: bool
    steps: int
    final_battery: int
    total_reward: float
    pickup_pos: Tuple[int, int]
    dropoff_pos: Tuple[int, int]
    start_pos: Tuple[int, int]


def run_agent_episodes(agent, agent_name: str, num_episodes: int = 50, max_steps: int = 100) -> Dict:
    """
    Run N episodes of an agent and collect comprehensive statistics.
    
    Args:
        agent: Agent instance to test
        agent_name: Name of the agent for identification
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary containing detailed statistics and results
    """
    print(f"\n🔬 Running {num_episodes} episodes for {agent_name} Agent")
    print("=" * 55)
    
    episode_results = []
    all_steps = []
    all_batteries = []
    all_rewards = []
    success_episodes = []
    failed_episodes = []
    
    for episode in range(num_episodes):
        if episode % 10 == 0:
            print(f"   Episode {episode + 1}/{num_episodes}...")
        
        # Initialize fresh agent for each episode
        if agent_name == "Greedy":
            agent = GreedyManhattanAgent()
            # Disable verbose output for batch runs
            agent.verbose = False
        elif agent_name == "Reflex":
            agent = SimpleReflexAgent()
        
        # Create environment
        env = WarehouseEnv(max_steps=max_steps)
        obs = env.reset(randomize=True)
        
        # Random start position
        start_pos = env._random_empty_cell()
        env.state.robot_pos = start_pos
        obs = env._observe()
        
        # Run episode
        step_count = 0
        total_reward = 0
        
        while step_count < max_steps:
            action = agent.act(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Collect episode statistics
        episode_result = {
            'episode': episode + 1,
            'success': terminated,
            'steps': step_count,
            'final_battery': obs['battery'],
            'total_reward': total_reward,
            'truncated': truncated
        }
        
        episode_results.append(episode_result)
        all_steps.append(step_count)
        all_batteries.append(obs['battery'])
        all_rewards.append(total_reward)
        
        if terminated:
            success_episodes.append(episode_result)
        else:
            failed_episodes.append(episode_result)
    
    # Calculate comprehensive statistics
    success_rate = len(success_episodes) / num_episodes * 100
    successful_steps = [ep['steps'] for ep in success_episodes] if success_episodes else [0]
    successful_batteries = [ep['final_battery'] for ep in success_episodes] if success_episodes else [0]
    successful_rewards = [ep['total_reward'] for ep in success_episodes] if success_episodes else [0]
    
    statistics = {
        'agent_name': agent_name,
        'num_episodes': num_episodes,
        'success_rate': success_rate,
        'successful_episodes': len(success_episodes),
        'failed_episodes': len(failed_episodes),
        
        # Episode length statistics
        'all_steps': all_steps,
        'successful_steps': successful_steps,
        'mean_steps_all': np.mean(all_steps),
        'median_steps_all': np.median(all_steps),
        'std_steps_all': np.std(all_steps),
        'mean_steps_successful': np.mean(successful_steps) if successful_steps else 0,
        'median_steps_successful': np.median(successful_steps) if successful_steps else 0,
        
        # Battery statistics
        'all_batteries': all_batteries,
        'successful_batteries': successful_batteries,
        'mean_battery_all': np.mean(all_batteries),
        'median_battery_all': np.median(all_batteries),
        'std_battery_all': np.std(all_batteries),
        'mean_battery_successful': np.mean(successful_batteries) if successful_batteries else 0,
        
        # Reward statistics
        'all_rewards': all_rewards,
        'successful_rewards': successful_rewards,
        'mean_reward_all': np.mean(all_rewards),
        'median_reward_all': np.median(all_rewards),
        'std_reward_all': np.std(all_rewards),
        'mean_reward_successful': np.mean(successful_rewards) if successful_rewards else 0,
        
        # Raw episode data
        'episode_results': episode_results,
        'success_episodes': success_episodes,
        'failed_episodes': failed_episodes
    }
    
    # Print summary
    print(f"\n📊 {agent_name} Agent Results:")
    print(f"   Success Rate: {success_rate:.1f}% ({len(success_episodes)}/{num_episodes})")
    print(f"   Mean Steps: {statistics['mean_steps_all']:.1f} (±{statistics['std_steps_all']:.1f})")
    print(f"   Mean Battery: {statistics['mean_battery_all']:.1f} (±{statistics['std_battery_all']:.1f})")
    print(f"   Mean Reward: {statistics['mean_reward_all']:.2f} (±{statistics['std_reward_all']:.2f})")
    
    return statistics


class AgentComparison:
    """
    Class to manage comparison between different agents with identical setups
    """
    
    def __init__(self, max_steps: int = 100, num_episodes: int = 50):
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.results: List[EpisodeResult] = []
        
        # Initialize agents
        self.greedy_agent = GreedyManhattanAgent()
        self.reflex_agent = SimpleReflexAgent()
        
        # Episode configurations to ensure identical setups
        self.episode_configs: List[Dict] = []
        
    def generate_episode_configurations(self) -> None:
        """Pre-generate all episode configurations for consistency"""
        print("🎯 Generating episode configurations for consistent comparison...")
        
        for episode_id in range(self.num_episodes):
            # Create a temporary environment to generate configuration
            env = WarehouseEnv(max_steps=self.max_steps)
            obs = env.reset(randomize=True)
            start_pos = env._random_empty_cell()
            
            config = {
                'episode_id': episode_id,
                'pickup_pos': obs['pickup_pos'],
                'dropoff_pos': obs['dropoff_pos'],
                'start_pos': start_pos,
                'grid': env.grid.copy(),
                'seed': random.randint(0, 1000000)
            }
            self.episode_configs.append(config)
        
        print(f"✅ Generated {len(self.episode_configs)} episode configurations")
    
    def run_single_agent_episode(self, agent, agent_name: str, config: Dict) -> EpisodeResult:
        """Run a single episode for one agent using the specified configuration"""
        
        # Set random seed for consistency
        random.seed(config['seed'])
        
        # Create environment with specific grid
        env = WarehouseEnv(grid=config['grid'], max_steps=self.max_steps)
        
        # Set specific pickup/dropoff positions by modifying the grid
        grid_rows = [list(row) for row in env.grid]
        
        # Clear old P and D positions
        for r in range(len(grid_rows)):
            for c in range(len(grid_rows[r])):
                if grid_rows[r][c] in ['P', 'D']:
                    grid_rows[r][c] = '.'
        
        # Set new positions
        pr, pc = config['pickup_pos']
        dr, dc = config['dropoff_pos']
        grid_rows[pr][pc] = 'P'
        grid_rows[dr][dc] = 'D'
        env.grid = [''.join(row) for row in grid_rows]
        
        # Reset environment and set robot position
        obs = env.reset(randomize=False)
        env.state.robot_pos = config['start_pos']
        obs = env._observe()
        
        # Run episode
        step_count = 0
        total_reward = 0
        
        while step_count < self.max_steps:
            action = agent.act(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Create result
        result = EpisodeResult(
            agent_name=agent_name,
            episode_id=config['episode_id'],
            success=terminated,
            steps=step_count,
            final_battery=obs['battery'],
            total_reward=total_reward,
            pickup_pos=config['pickup_pos'],
            dropoff_pos=config['dropoff_pos'],
            start_pos=config['start_pos']
        )
        
        return result
    
    def run_comparison_episodes(self) -> None:
        """Run all episodes for both agents with identical configurations"""
        print(f"\n🔄 Running {self.num_episodes} episodes for both agents...")
        print("=" * 60)
        
        for i, config in enumerate(self.episode_configs):
            print(f"\n📋 Episode {i+1}/{self.num_episodes}")
            print(f"   Pickup: {config['pickup_pos']}, Dropoff: {config['dropoff_pos']}")
            print(f"   Start: {config['start_pos']}")
            
            # Run Greedy Agent
            greedy_result = self.run_single_agent_episode(
                self.greedy_agent, "Greedy", config
            )
            self.results.append(greedy_result)
            
            # Reset agent state for fair comparison
            self.greedy_agent = GreedyManhattanAgent()  # Fresh instance
            
            # Run Reflex Agent
            reflex_result = self.run_single_agent_episode(
                self.reflex_agent, "Reflex", config
            )
            self.results.append(reflex_result)
            
            # Log episode results
            print(f"   🤖 Greedy: {'✅' if greedy_result.success else '❌'} "
                  f"({greedy_result.steps} steps, {greedy_result.final_battery} battery)")
            print(f"   🧠 Reflex: {'✅' if reflex_result.success else '❌'} "
                  f"({reflex_result.steps} steps, {reflex_result.final_battery} battery)")
    
    def analyze_results(self) -> Dict:
        """Analyze results and compute statistics"""
        print(f"\n📊 Analyzing Results...")
        print("=" * 30)
        
        # Separate results by agent
        greedy_results = [r for r in self.results if r.agent_name == "Greedy"]
        reflex_results = [r for r in self.results if r.agent_name == "Reflex"]
        
        def compute_stats(results: List[EpisodeResult], agent_name: str) -> Dict:
            """Compute statistics for one agent"""
            total_episodes = len(results)
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            success_rate = len(successful) / total_episodes * 100 if total_episodes > 0 else 0
            
            # Episode lengths
            all_steps = [r.steps for r in results]
            successful_steps = [r.steps for r in successful] if successful else [0]
            
            # Battery levels
            all_batteries = [r.final_battery for r in results]
            successful_batteries = [r.final_battery for r in successful] if successful else [0]
            
            # Rewards
            all_rewards = [r.total_reward for r in results]
            successful_rewards = [r.total_reward for r in successful] if successful else [0]
            
            return {
                'agent_name': agent_name,
                'total_episodes': total_episodes,
                'successful_episodes': len(successful),
                'failed_episodes': len(failed),
                'success_rate': success_rate,
                'all_steps': all_steps,
                'successful_steps': successful_steps,
                'mean_steps_all': np.mean(all_steps) if all_steps else 0,
                'median_steps_all': np.median(all_steps) if all_steps else 0,
                'mean_steps_successful': np.mean(successful_steps) if successful_steps else 0,
                'median_steps_successful': np.median(successful_steps) if successful_steps else 0,
                'all_batteries': all_batteries,
                'successful_batteries': successful_batteries,
                'mean_battery': np.mean(all_batteries) if all_batteries else 0,
                'all_rewards': all_rewards,
                'mean_reward': np.mean(all_rewards) if all_rewards else 0
            }
        
        greedy_stats = compute_stats(greedy_results, "Greedy")
        reflex_stats = compute_stats(reflex_results, "Reflex")
        
        # Print summary
        print(f"\n🤖 Greedy Agent Results:")
        print(f"   Success Rate: {greedy_stats['success_rate']:.1f}% ({greedy_stats['successful_episodes']}/{greedy_stats['total_episodes']})")
        print(f"   Mean Steps (All): {greedy_stats['mean_steps_all']:.1f}")
        print(f"   Mean Steps (Success): {greedy_stats['mean_steps_successful']:.1f}")
        print(f"   Mean Battery: {greedy_stats['mean_battery']:.1f}")
        print(f"   Mean Reward: {greedy_stats['mean_reward']:.2f}")
        
        print(f"\n🧠 Reflex Agent Results:")
        print(f"   Success Rate: {reflex_stats['success_rate']:.1f}% ({reflex_stats['successful_episodes']}/{reflex_stats['total_episodes']})")
        print(f"   Mean Steps (All): {reflex_stats['mean_steps_all']:.1f}")
        print(f"   Mean Steps (Success): {reflex_stats['mean_steps_successful']:.1f}")
        print(f"   Mean Battery: {reflex_stats['mean_battery']:.1f}")
        print(f"   Mean Reward: {reflex_stats['mean_reward']:.2f}")
        
        return {
            'greedy': greedy_stats,
            'reflex': reflex_stats,
            'comparison': {
                'success_rate_diff': greedy_stats['success_rate'] - reflex_stats['success_rate'],
                'mean_steps_diff': greedy_stats['mean_steps_all'] - reflex_stats['mean_steps_all'],
                'mean_battery_diff': greedy_stats['mean_battery'] - reflex_stats['mean_battery'],
                'mean_reward_diff': greedy_stats['mean_reward'] - reflex_stats['mean_reward']
            }
        }
    
    def create_individual_agent_visualization(self, agent_stats: Dict) -> None:
        """Create detailed visualization for a single agent's performance"""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            # Try different backends
            backends = ['TkAgg', 'Qt5Agg', 'MacOSX']
            for backend in backends:
                try:
                    matplotlib.use(backend)
                    break
                except:
                    continue
            plt.ion()  # Turn on interactive mode
            print(f"✅ Matplotlib loaded with {matplotlib.get_backend()} backend")
        except ImportError:
            print("❌ matplotlib not available - install with: pip install matplotlib")
            return
        except Exception as e:
            print(f"❌ Error setting up matplotlib: {e}")
            return
        
        agent_name = agent_stats['agent_name']
        print(f"\n📊 Creating detailed visualization for {agent_name} Agent...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{agent_name} Agent Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Success Rate Pie Chart
        ax = axes[0, 0]
        success_count = agent_stats['successful_episodes']
        failure_count = len(agent_stats['failed_episodes'])  # Length of the failed episodes list
        
        # Ensure values are integers
        success_count = int(success_count) if success_count is not None else 0
        failure_count = int(failure_count) if failure_count is not None else 0
        
        labels = ['Success', 'Failure']
        sizes = [success_count, failure_count]
        colors = ['#28a745', '#dc3545']
        
        # Only create pie chart if we have valid data
        if success_count + failure_count > 0:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Success Rate: {agent_stats["success_rate"]:.1f}%')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title('No Data')
        
        # 2. Episode Length Distribution
        ax = axes[0, 1]
        ax.hist(agent_stats['all_steps'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(agent_stats['mean_steps_all'], color='red', linestyle='--', 
                  label=f'Mean: {agent_stats["mean_steps_all"]:.1f}')
        ax.axvline(agent_stats['median_steps_all'], color='orange', linestyle='--', 
                  label=f'Median: {agent_stats["median_steps_all"]:.1f}')
        ax.set_xlabel('Episode Length (Steps)')
        ax.set_ylabel('Frequency')
        ax.set_title('Episode Length Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Battery Level Distribution
        ax = axes[0, 2]
        ax.hist(agent_stats['all_batteries'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.axvline(agent_stats['mean_battery_all'], color='red', linestyle='--', 
                  label=f'Mean: {agent_stats["mean_battery_all"]:.1f}')
        ax.axvline(agent_stats['median_battery_all'], color='orange', linestyle='--', 
                  label=f'Median: {agent_stats["median_battery_all"]:.1f}')
        ax.set_xlabel('Final Battery Level')
        ax.set_ylabel('Frequency')
        ax.set_title('Final Battery Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Reward Distribution
        ax = axes[1, 0]
        ax.hist(agent_stats['all_rewards'], bins=15, alpha=0.7, color='plum', edgecolor='black')
        ax.axvline(agent_stats['mean_reward_all'], color='red', linestyle='--', 
                  label=f'Mean: {agent_stats["mean_reward_all"]:.2f}')
        ax.axvline(agent_stats['median_reward_all'], color='orange', linestyle='--', 
                  label=f'Median: {agent_stats["median_reward_all"]:.2f}')
        ax.set_xlabel('Total Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Success vs Failure Comparison (Steps)
        ax = axes[1, 1]
        if len(agent_stats['successful_steps']) > 0 and len(agent_stats['failed_episodes']) > 0:
            data_to_plot = [agent_stats['successful_steps'], 
                           [ep['steps'] for ep in agent_stats['failed_episodes']]]
            labels = ['Successful', 'Failed']
            box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            box_plot['boxes'][0].set_facecolor('#28a745')
            if len(box_plot['boxes']) > 1:
                box_plot['boxes'][1].set_facecolor('#dc3545')
        else:
            ax.boxplot([agent_stats['all_steps']], labels=[agent_name])
        ax.set_ylabel('Episode Length (Steps)')
        ax.set_title('Steps: Success vs Failure')
        ax.grid(True, alpha=0.3)
        
        # 6. Performance Over Time
        ax = axes[1, 2]
        episodes = list(range(1, len(agent_stats['episode_results']) + 1))
        success_markers = [ep['episode'] for ep in agent_stats['success_episodes']]
        success_steps = [ep['steps'] for ep in agent_stats['success_episodes']]
        failure_markers = [ep['episode'] for ep in agent_stats['failed_episodes']]
        failure_steps = [ep['steps'] for ep in agent_stats['failed_episodes']]
        
        if success_markers:
            ax.scatter(success_markers, success_steps, color='green', alpha=0.6, s=30, label='Success')
        if failure_markers:
            ax.scatter(failure_markers, failure_steps, color='red', alpha=0.6, s=30, label='Failure')
        
        ax.set_xlabel('Episode Number')
        ax.set_ylabel('Steps Taken')
        ax.set_title('Performance Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        try:
            plt.show(block=False)  # Non-blocking show
            plt.pause(2)  # Pause to ensure display
        except Exception as e:
            print(f"⚠️ Display error: {e}")
        
        print(f"✅ Individual visualization for {agent_name} Agent completed!")
    
    def create_visualizations(self, analysis: Dict) -> None:
        """Create comprehensive visualizations comparing both agents"""
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Use TkAgg backend for display
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.ion()  # Turn on interactive mode
        except ImportError:
            print("❌ matplotlib/seaborn not available for visualizations")
            return
        except Exception as e:
            print(f"❌ Error setting up matplotlib: {e}")
            return
        
        print(f"\n📈 Creating Visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Success Rate Comparison (Bar Chart)
        ax1 = plt.subplot(2, 3, 1)
        agents = ['Greedy', 'Reflex']
        success_rates = [analysis['greedy']['success_rate'], analysis['reflex']['success_rate']]
        colors = ['#2E86AB', '#A23B72']
        bars = ax1.bar(agents, success_rates, color=colors, alpha=0.7)
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate Comparison')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{value:.1f}%', ha='center', va='bottom')
        
        # 2. Episode Length Box Plot
        ax2 = plt.subplot(2, 3, 2)
        step_data = [analysis['greedy']['all_steps'], analysis['reflex']['all_steps']]
        box_plot = ax2.boxplot(step_data, labels=agents, patch_artist=True)
        box_plot['boxes'][0].set_facecolor(colors[0])
        box_plot['boxes'][1].set_facecolor(colors[1])
        ax2.set_ylabel('Episode Length (Steps)')
        ax2.set_title('Episode Length Distribution')
        
        # 3. Battery Level Histograms
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(analysis['greedy']['all_batteries'], bins=15, alpha=0.7, 
                 color=colors[0], label='Greedy', density=True)
        ax3.hist(analysis['reflex']['all_batteries'], bins=15, alpha=0.7, 
                 color=colors[1], label='Reflex', density=True)
        ax3.set_xlabel('Final Battery Level')
        ax3.set_ylabel('Density')
        ax3.set_title('Final Battery Level Distribution')
        ax3.legend()
        
        # 4. Success Rate by Episode (Running Average)
        ax4 = plt.subplot(2, 3, 4)
        
        # Calculate running success rates
        greedy_results = [r for r in self.results if r.agent_name == "Greedy"]
        reflex_results = [r for r in self.results if r.agent_name == "Reflex"]
        
        def running_success_rate(results, window=5):
            rates = []
            for i in range(len(results)):
                start = max(0, i - window + 1)
                window_results = results[start:i+1]
                rate = sum(1 for r in window_results if r.success) / len(window_results) * 100
                rates.append(rate)
            return rates
        
        episodes = list(range(1, self.num_episodes + 1))
        greedy_running = running_success_rate(greedy_results)
        reflex_running = running_success_rate(reflex_results)
        
        ax4.plot(episodes, greedy_running, color=colors[0], label='Greedy', linewidth=2)
        ax4.plot(episodes, reflex_running, color=colors[1], label='Reflex', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Running Success Rate (%)')
        ax4.set_title('Running Success Rate (5-episode window)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Mean Reward Comparison
        ax5 = plt.subplot(2, 3, 5)
        mean_rewards = [analysis['greedy']['mean_reward'], analysis['reflex']['mean_reward']]
        bars = ax5.bar(agents, mean_rewards, color=colors, alpha=0.7)
        ax5.set_ylabel('Mean Total Reward')
        ax5.set_title('Average Total Reward')
        
        # Add value labels on bars
        for bar, value in zip(bars, mean_rewards):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(mean_rewards) * 0.01),
                     f'{value:.2f}', ha='center', va='bottom')
        
        # 6. Performance Summary Table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Create summary table data
        table_data = [
            ['Metric', 'Greedy', 'Reflex', 'Difference'],
            ['Success Rate (%)', f"{analysis['greedy']['success_rate']:.1f}", 
             f"{analysis['reflex']['success_rate']:.1f}", 
             f"{analysis['comparison']['success_rate_diff']:+.1f}"],
            ['Mean Steps', f"{analysis['greedy']['mean_steps_all']:.1f}", 
             f"{analysis['reflex']['mean_steps_all']:.1f}", 
             f"{analysis['comparison']['mean_steps_diff']:+.1f}"],
            ['Mean Battery', f"{analysis['greedy']['mean_battery']:.1f}", 
             f"{analysis['reflex']['mean_battery']:.1f}", 
             f"{analysis['comparison']['mean_battery_diff']:+.1f}"],
            ['Mean Reward', f"{analysis['greedy']['mean_reward']:.2f}", 
             f"{analysis['reflex']['mean_reward']:.2f}", 
             f"{analysis['comparison']['mean_reward_diff']:+.2f}"]
        ]
        
        table = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                          cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax6.set_title('Performance Summary', pad=20)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Visualizations created successfully!")
    
    def create_summary_visualization(self, greedy_stats: Dict, reflex_stats: Dict) -> None:
        """
        Create a focused summary visualization comparing both agents.
        Shows: (1) Success rates bar chart, (2) Episode lengths box plot, (3) Battery histograms
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            # Try different backends
            backends = ['TkAgg', 'Qt5Agg', 'MacOSX']
            for backend in backends:
                try:
                    matplotlib.use(backend)
                    break
                except:
                    continue
            plt.ion()  # Turn on interactive mode
            print(f"✅ Matplotlib loaded with {matplotlib.get_backend()} backend")
        except ImportError:
            print("❌ matplotlib not available - install with: pip install matplotlib")
            return
        except Exception as e:
            print(f"❌ Error setting up matplotlib: {e}")
            return
        
        print(f"\n📈 Creating Summary Visualization...")
        
        # Create figure with 3 subplots as requested
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Agent Comparison Summary', fontsize=16, fontweight='bold')
        
        # Colors for consistency
        colors = {'Greedy': '#2E86AB', 'Reflex': '#A23B72'}
        
        # 1. Bar Chart of Success Rates
        agents = ['Greedy', 'Reflex']
        success_rates = [greedy_stats['success_rate'], reflex_stats['success_rate']]
        bars = ax1.bar(agents, success_rates, color=[colors['Greedy'], colors['Reflex']], alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate Comparison')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add statistical significance indicator
        diff = abs(greedy_stats['success_rate'] - reflex_stats['success_rate'])
        if diff > 10:
            winner = 'Greedy' if greedy_stats['success_rate'] > reflex_stats['success_rate'] else 'Reflex'
            ax1.text(0.5, 95, f'{winner} significantly better', ha='center', 
                    transform=ax1.transData, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 2. Box Plots of Episode Lengths
        episode_data = [greedy_stats['all_steps'], reflex_stats['all_steps']]
        box_plot = ax2.boxplot(episode_data, tick_labels=agents, patch_artist=True)
        box_plot['boxes'][0].set_facecolor(colors['Greedy'])
        box_plot['boxes'][1].set_facecolor(colors['Reflex'])
        
        # Add mean markers
        means = [greedy_stats['mean_steps_all'], reflex_stats['mean_steps_all']]
        ax2.scatter([1, 2], means, color='red', marker='D', s=50, zorder=10, label='Mean')
        
        ax2.set_ylabel('Episode Length (Steps)')
        ax2.set_title('Episode Length Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        # Add efficiency comparison
        if greedy_stats['mean_steps_all'] < reflex_stats['mean_steps_all']:
            efficiency_text = f"Greedy {reflex_stats['mean_steps_all'] - greedy_stats['mean_steps_all']:.1f} steps fewer"
        else:
            efficiency_text = f"Reflex {greedy_stats['mean_steps_all'] - reflex_stats['mean_steps_all']:.1f} steps fewer"
        
        ax2.text(0.5, 0.95, efficiency_text, transform=ax2.transAxes, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # 3. Histograms of Final Battery Levels
        ax3.hist(greedy_stats['all_batteries'], bins=15, alpha=0.6, 
                color=colors['Greedy'], label='Greedy', density=True)
        ax3.hist(reflex_stats['all_batteries'], bins=15, alpha=0.6, 
                color=colors['Reflex'], label='Reflex', density=True)
        
        # Add mean lines
        ax3.axvline(greedy_stats['mean_battery_all'], color=colors['Greedy'], 
                   linestyle='--', linewidth=2, alpha=0.8)
        ax3.axvline(reflex_stats['mean_battery_all'], color=colors['Reflex'], 
                   linestyle='--', linewidth=2, alpha=0.8)
        
        ax3.set_xlabel('Final Battery Level')
        ax3.set_ylabel('Density')
        ax3.set_title('Final Battery Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add battery efficiency comparison
        battery_diff = greedy_stats['mean_battery_all'] - reflex_stats['mean_battery_all']
        if abs(battery_diff) > 5:
            better_agent = 'Greedy' if battery_diff > 0 else 'Reflex'
            ax3.text(0.5, 0.95, f'{better_agent} conserves +{abs(battery_diff):.1f} battery', 
                    transform=ax3.transAxes, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        try:
            plt.show(block=False)  # Non-blocking show
            plt.pause(3)  # Pause to ensure display
        except Exception as e:
            print(f"⚠️ Display error: {e}")
        
        print("✅ Summary visualization completed!")
    
    def create_side_by_side_demo(self, episode_config: Dict) -> None:
        """Create a side-by-side animation demo of both agents"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib import animation
        except ImportError:
            print("❌ matplotlib not available for side-by-side demo")
            return
        
        print(f"\n🎬 Creating Side-by-Side Demo Animation...")
        
        # Run both agents on the same episode and collect frames
        def run_agent_for_demo(agent, agent_name: str) -> Tuple[List, Dict]:
            # Set random seed for consistency
            random.seed(episode_config['seed'])
            
            # Create environment
            env = WarehouseEnv(grid=episode_config['grid'], max_steps=self.max_steps)
            
            # Set positions
            grid_rows = [list(row) for row in env.grid]
            for r in range(len(grid_rows)):
                for c in range(len(grid_rows[r])):
                    if grid_rows[r][c] in ['P', 'D']:
                        grid_rows[r][c] = '.'
            
            pr, pc = episode_config['pickup_pos']
            dr, dc = episode_config['dropoff_pos']
            grid_rows[pr][pc] = 'P'
            grid_rows[dr][dc] = 'D'
            env.grid = [''.join(row) for row in grid_rows]
            
            obs = env.reset(randomize=False)
            env.state.robot_pos = episode_config['start_pos']
            obs = env._observe()
            
            frames = []
            metrics = {
                "rewards": [],
                "battery": [],
                "dist_pickup": [],
                "dist_dropoff": []
            }
            
            step_count = 0
            
            def manhattan_distance(pos1, pos2):
                return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
            
            while step_count < min(50, self.max_steps):  # Limit for demo
                frames.append(env.render_grid())
                
                current_pos = obs["robot_pos"]
                pickup_pos = obs["pickup_pos"]
                dropoff_pos = obs["dropoff_pos"]
                
                dist_pickup = manhattan_distance(current_pos, pickup_pos) if pickup_pos else 0
                dist_dropoff = manhattan_distance(current_pos, dropoff_pos) if dropoff_pos else 0
                
                metrics["battery"].append(obs["battery"])
                metrics["dist_pickup"].append(dist_pickup)
                metrics["dist_dropoff"].append(dist_dropoff)
                
                action = agent.act(obs, env)
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                
                metrics["rewards"].append(reward)
                
                if terminated or truncated:
                    break
            
            # Add final frame
            frames.append(env.render_grid())
            metrics["battery"].append(obs["battery"])
            metrics["dist_pickup"].append(dist_pickup)
            metrics["dist_dropoff"].append(dist_dropoff)
            metrics["rewards"].append(0.0)
            
            return frames, metrics
        
        # Get frames for both agents
        greedy_frames, greedy_metrics = run_agent_for_demo(GreedyManhattanAgent(), "Greedy")
        reflex_frames, reflex_metrics = run_agent_for_demo(SimpleReflexAgent(), "Reflex")
        
        # Ensure both have the same number of frames
        max_frames = max(len(greedy_frames), len(reflex_frames))
        while len(greedy_frames) < max_frames:
            greedy_frames.append(greedy_frames[-1])
            for key in greedy_metrics:
                greedy_metrics[key].append(greedy_metrics[key][-1])
        while len(reflex_frames) < max_frames:
            reflex_frames.append(reflex_frames[-1])
            for key in reflex_metrics:
                reflex_metrics[key].append(reflex_metrics[key][-1])
        
        # Create side-by-side animation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Setup left side (Greedy)
        ax1.set_title("Greedy Manhattan Agent", fontsize=14, fontweight='bold')
        ax1.set_axis_off()
        im1 = ax1.imshow(_grid_to_rgb(greedy_frames[0]), interpolation="nearest")
        
        # Setup right side (Reflex)
        ax2.set_title("Simple Reflex Agent", fontsize=14, fontweight='bold')
        ax2.set_axis_off()
        im2 = ax2.imshow(_grid_to_rgb(reflex_frames[0]), interpolation="nearest")
        
        # Add step counter
        step_text = fig.suptitle("Step 0", fontsize=16)
        
        def update(frame):
            im1.set_data(_grid_to_rgb(greedy_frames[frame]))
            im2.set_data(_grid_to_rgb(reflex_frames[frame]))
            step_text.set_text(f"Step {frame}")
            return [im1, im2, step_text]
        
        anim = animation.FuncAnimation(
            fig, update, frames=max_frames, interval=600, blit=False, repeat=True
        )
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Side-by-side demo created successfully!")
        return anim
    
    def save_results(self, analysis: Dict, filename: str = "agent_comparison_results.json") -> None:
        """Save detailed results to JSON file"""
        output_data = {
            'metadata': {
                'num_episodes': self.num_episodes,
                'max_steps': self.max_steps,
                'timestamp': str(np.datetime64('now'))
            },
            'results': [asdict(result) for result in self.results],
            'analysis': analysis
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Results saved to {filename}")

    def create_comprehensive_visualization(self, greedy_stats, reflex_stats):
        """Create a comprehensive single-screen visualization with all results"""
        try:
            if not self._setup_plotting():
                return
            
            print(f"\n📈 Creating Comprehensive Single-Screen Visualization...")
            
            # Create a large figure with custom grid layout
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(24, 18))
            fig.suptitle('🤖 Complete Agent Performance Analysis', fontsize=20, fontweight='bold', y=0.98)
            
            # Define grid: 3 rows with different column arrangements
            # Row 1: Greedy Agent (6 subplots)
            # Row 2: Reflex Agent (6 subplots)  
            # Row 3: Comparison Summary (3 subplots, centered)
            
            # Row 1: Greedy Agent Analysis (5 subplots)
            greedy_axes = []
            for i in range(5):
                ax = plt.subplot2grid((3, 6), (0, i), fig=fig)
                greedy_axes.append(ax)
            
            # Row 2: Reflex Agent Analysis (5 subplots)
            reflex_axes = []
            for i in range(5):
                ax = plt.subplot2grid((3, 6), (1, i), fig=fig)
                reflex_axes.append(ax)
            
            # Row 3: Summary Comparison (3 subplots, centered)
            summary_axes = []
            for i in range(3):
                ax = plt.subplot2grid((3, 6), (2, i + 1), colspan=1, fig=fig)
                summary_axes.append(ax)
            
            # Create individual agent visualizations
            self._create_agent_subplot_analysis(greedy_axes, greedy_stats, "🎯 Greedy Agent")
            self._create_agent_subplot_analysis(reflex_axes, reflex_stats, "🧠 Reflex Agent")
            
            # Create summary comparison
            self._create_summary_subplots(summary_axes, greedy_stats, reflex_stats)
            
            # Adjust layout and display
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for main title
            
            # Keep the window open
            print(f"\n✅ Comprehensive visualization created!")
            print(f"📺 Displaying all results in a single window...")
            print(f"💡 Close the window or press Ctrl+C to exit")
            
            plt.show(block=True)  # This keeps the window open
            
        except Exception as e:
            print(f"❌ Error creating comprehensive visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _setup_plotting(self):
        """Setup matplotlib with fallback backends"""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            # Try different backends
            backends = ['TkAgg', 'Qt5Agg', 'MacOSX']
            for backend in backends:
                try:
                    matplotlib.use(backend)
                    break
                except:
                    continue
            plt.ion()  # Turn on interactive mode
            print(f"✅ Matplotlib loaded with {matplotlib.get_backend()} backend")
            return True
        except Exception as e:
            print(f"❌ matplotlib/seaborn not available for visualizations: {e}")
            return False
    
    def _create_agent_subplot_analysis(self, axes, agent_stats, title_prefix):
        """Create 6 subplots for individual agent analysis"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        agent_name = title_prefix.split()[-1]
        
        # 1. Success Rate Pie Chart
        ax = axes[0]
        success_count = agent_stats['successful_episodes']
        failure_count = len(agent_stats['failed_episodes'])
        
        if success_count + failure_count > 0:
            sizes = [success_count, failure_count]
            labels = ['Success', 'Failure']
            colors = ['#28a745', '#dc3545']
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'{title_prefix}\\nSuccess Rate: {agent_stats["success_rate"]:.1f}%', fontsize=10)
        
        # 2. Episode Length Distribution
        ax = axes[1]
        ax.hist(agent_stats['all_steps'], bins=12, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(agent_stats['mean_steps_all'], color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Episode Length (Steps)', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'Episode Length\\nMean: {agent_stats["mean_steps_all"]:.1f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 3. Battery Level Distribution
        ax = axes[2]
        ax.hist(agent_stats['all_batteries'], bins=12, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.axvline(agent_stats['mean_battery_all'], color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Final Battery Level', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'Battery Levels (Mean: {agent_stats["mean_battery_all"]:.1f})', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 3. Reward Distribution
        ax = axes[2]
        ax.hist(agent_stats['all_rewards'], bins=12, alpha=0.7, color='orange', edgecolor='black')
        ax.axvline(agent_stats['mean_reward_all'], color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Total Reward', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'Rewards (Mean: {agent_stats["mean_reward_all"]:.2f})', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 4. Episode Performance Trend
        ax = axes[3]
        episodes = list(range(1, len(agent_stats['all_steps']) + 1))
        ax.scatter(episodes, agent_stats['all_steps'], alpha=0.6, s=30)
        ax.plot(episodes, agent_stats['all_steps'], alpha=0.3, color='gray')
        ax.set_xlabel('Episode Number', fontsize=9)
        ax.set_ylabel('Steps Taken', fontsize=9)
        ax.set_title('Performance Trend', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 5. Success vs Failure Comparison
        ax = axes[4]
        if len(agent_stats['successful_steps']) > 0 and len(agent_stats['failed_episodes']) > 0:
            data_to_plot = [agent_stats['successful_steps'], 
                           [ep['steps'] for ep in agent_stats['failed_episodes']]]
            box_plot = ax.boxplot(data_to_plot, tick_labels=['Success', 'Failure'], patch_artist=True)
            box_plot['boxes'][0].set_facecolor('#28a745')
            if len(box_plot['boxes']) > 1:
                box_plot['boxes'][1].set_facecolor('#dc3545')
            ax.set_ylabel('Steps', fontsize=9)
            ax.set_title('Success vs Failure Step Comparison', fontsize=10)
            ax.grid(True, alpha=0.3)
        elif len(agent_stats['successful_steps']) > 0:
            # Only successful episodes
            ax.hist(agent_stats['successful_steps'], bins=10, alpha=0.7, color='#28a745', edgecolor='black')
            ax.set_xlabel('Steps', fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            ax.set_title('Successful Episodes Only', fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Successful Episodes', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Success vs Failure', fontsize=10)
        ax.plot(episodes, agent_stats['all_steps'], alpha=0.3, color='gray')
        ax.set_xlabel('Episode Number', fontsize=9)
        ax.set_ylabel('Steps Taken', fontsize=9)
        ax.set_title('Performance Trend', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _create_summary_subplots(self, axes, greedy_stats, reflex_stats):
        """Create 3 summary comparison subplots"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 1. Success Rate Bar Chart
        ax = axes[0]
        agents = ['Greedy', 'Reflex']
        success_rates = [greedy_stats['success_rate'], reflex_stats['success_rate']]
        
        bars = ax.bar(agents, success_rates, color=['#2E8B57', '#4682B4'], alpha=0.8, edgecolor='black')
        ax.set_ylabel('Success Rate (%)', fontsize=10)
        ax.set_title('🏆 Success Rate\\nComparison', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Episode Length Box Plot
        ax = axes[1]
        greedy_steps = greedy_stats['all_steps']
        reflex_steps = reflex_stats['all_steps']
        
        box_data = [greedy_steps, reflex_steps]
        box_plot = ax.boxplot(box_data, tick_labels=['Greedy', 'Reflex'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('#2E8B57')
        if len(box_plot['boxes']) > 1:
            box_plot['boxes'][1].set_facecolor('#4682B4')
        
        ax.set_ylabel('Episode Length (Steps)', fontsize=10)
        ax.set_title('⚡ Episode Length\\nDistribution', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. Final Battery Level Histograms
        ax = axes[2]
        greedy_battery = greedy_stats['all_batteries']
        reflex_battery = reflex_stats['all_batteries']
        
        # Create overlapping histograms
        ax.hist(greedy_battery, bins=10, alpha=0.7, label='Greedy', color='#2E8B57', edgecolor='black')
        ax.hist(reflex_battery, bins=10, alpha=0.7, label='Reflex', color='#4682B4', edgecolor='black')
        
        ax.set_xlabel('Final Battery Level', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('🔋 Battery Level\\nDistribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)


def main():
    """Main execution function"""
    print("🤖🧠 Agent Comparison: Greedy vs Reflex")
    print("=" * 50)
    
    # Parameters (moderate size for good results)
    num_episodes = 25  # Balanced number for good statistics
    max_steps = 75     # Reasonable episode length
    
    # Run individual agent analysis
    print("\n🔬 Running Individual Agent Analysis")
    print("=" * 45)
    
    greedy_stats = run_agent_episodes(
        agent=GreedyManhattanAgent(),
        agent_name="Greedy",
        num_episodes=num_episodes,
        max_steps=max_steps
    )
    
    reflex_stats = run_agent_episodes(
        agent=SimpleReflexAgent(),
        agent_name="Reflex", 
        num_episodes=num_episodes,
        max_steps=max_steps
    )
    
    # Create comprehensive single-screen visualization
    print("\n\n📊 Creating Comprehensive Visualization (All Results in One Screen)")
    print("=" * 70)
    comparison = AgentComparison()
    comparison.create_comprehensive_visualization(greedy_stats, reflex_stats)
    
    # Print final comparison summary
    print("\n\n🏆 Final Comparison Results:")
    print("=" * 35)
    print(f"Greedy Agent: {greedy_stats['success_rate']:.1f}% success, "
          f"{greedy_stats['mean_steps_all']:.1f} avg steps, "
          f"{greedy_stats['mean_battery_all']:.1f} avg battery")
    print(f"Reflex Agent: {reflex_stats['success_rate']:.1f}% success, "
          f"{reflex_stats['mean_steps_all']:.1f} avg steps, "
          f"{reflex_stats['mean_battery_all']:.1f} avg battery")
    
    success_diff = greedy_stats['success_rate'] - reflex_stats['success_rate']
    steps_diff = greedy_stats['mean_steps_all'] - reflex_stats['mean_steps_all']
    battery_diff = greedy_stats['mean_battery_all'] - reflex_stats['mean_battery_all']
    
    print(f"\n🔄 Differences (Greedy - Reflex):")
    print(f"   Success Rate: {success_diff:+.1f}%")
    print(f"   Average Steps: {steps_diff:+.1f}")
    print(f"   Average Battery: {battery_diff:+.1f}")
    
    # Save results to file
    import json
    results = {
        'greedy': greedy_stats,
        'reflex': reflex_stats,
        'comparison': {
            'success_rate_diff': success_diff,
            'steps_diff': steps_diff,
            'battery_diff': battery_diff
        }
    }
    
    # Remove non-serializable data for JSON
    for agent in ['greedy', 'reflex']:
        if 'episode_results' in results[agent]:
            del results[agent]['episode_results']
        if 'success_episodes' in results[agent]:
            del results[agent]['success_episodes']
        if 'failed_episodes' in results[agent]:
            del results[agent]['failed_episodes']
    
    with open('agent_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to agent_comparison_results.json")
    
    print(f"\n✅ Agent comparison completed successfully!")


if __name__ == "__main__":
    main()
