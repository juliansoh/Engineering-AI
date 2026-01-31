"""
Simple Reflex Agent for Warehouse Environment
============================================

This agent implements a simple reflex-based behavior:
1. If at pickup location and no item: pick
2. If at dropoff location and carrying item: drop
3. If carrying item: move toward dropoff (N/S/E/W based on relative position)
4. If not carrying item: move toward pickup (N/S/E/W based on relative position)
5. If no rule applies: choose a random valid action

Author: Created for EAI Chapter 1 - Problem 1-1
"""

import random
import sys
import os
from typing import List, Tuple, Dict

# Add the warehouse agent source directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../Warehouse_Agent/src'))

from warehouse_env import WarehouseEnv
from warehouse_viz import replay_animation


class SimpleReflexAgent:
    """
    Simple reflex agent that follows explicit condition-action rules based on current position,
    goal positions, and whether the robot carries an item.
    
    Uses 8-directional awareness with 4-directional movement (N, S, E, W only).
    
    Rules (in order of priority):
    1. If at pickup and no item -> PICK
    2. If at dropoff and carrying item -> DROP
    3-10. If carrying item and dropoff is [N/S/E/W/NE/NW/SE/SW] -> move appropriately
    11-18. If not carrying item and pickup is [N/S/E/W/NE/NW/SE/SW] -> move appropriately  
    19. Otherwise -> random valid move
    
    The agent includes randomized fallback to avoid loops when direct paths are blocked.
    """
    
    def __init__(self):
        # Simple reflex agents are stateless - no internal memory
        # But we add a tiny bit of randomization to break loops
        self.random_counter = 0
    
    def get_valid_moves(self, env: WarehouseEnv, current_pos: Tuple[int, int]) -> List[str]:
        """Get all valid movement actions from current position."""
        valid_moves = []
        
        for action, (dr, dc) in env.MOVE_DELTAS.items():
            r, c = current_pos
            nr, nc = r + dr, c + dc
            
            # Check if the move is valid (not into a wall)
            if not env._is_wall(nr, nc):
                valid_moves.append(action)
        
        return valid_moves
    
    def get_direction_to_goal(self, current_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> str:
        """Determine the 8-directional relationship between current position and goal."""
        curr_r, curr_c = current_pos
        goal_r, goal_c = goal_pos
        
        # Calculate directional differences
        dr = goal_r - curr_r
        dc = goal_c - curr_c
        
        # Determine 8-directional relationship
        if dr == 0 and dc == 0:
            return "HERE"  # Already at goal
        elif dr < 0 and dc == 0:
            return "N"   # Goal is North
        elif dr > 0 and dc == 0:
            return "S"   # Goal is South
        elif dr == 0 and dc > 0:
            return "E"   # Goal is East
        elif dr == 0 and dc < 0:
            return "W"   # Goal is West
        elif dr < 0 and dc > 0:
            return "NE"  # Goal is Northeast
        elif dr < 0 and dc < 0:
            return "NW"  # Goal is Northwest
        elif dr > 0 and dc > 0:
            return "SE"  # Goal is Southeast
        elif dr > 0 and dc < 0:
            return "SW"  # Goal is Southwest
        else:
            return "HERE"  # Fallback
    
    def act(self, obs: Dict, env: WarehouseEnv) -> str:
        """
        Simple reflex action selection based on explicit condition-action rules.
        """
        current_pos = obs["robot_pos"]
        has_item = obs["has_item"]
        pickup_pos = obs["pickup_pos"]
        dropoff_pos = obs["dropoff_pos"]
        
        # Increment counter for loop-breaking randomization
        self.random_counter += 1
        
        # Rule 1: If at pickup location and no item -> PICK
        if not has_item and pickup_pos == current_pos:
            return "PICK"
        
        # Rule 2: If at dropoff location and carrying item -> DROP
        if has_item and dropoff_pos == current_pos:
            return "DROP"
        
        # Get all valid moves for fallbacks
        valid_moves = self.get_valid_moves(env, current_pos)
        
        # Rules 3-10: If carrying item, move toward dropoff based on 8-directional relationship
        if has_item and dropoff_pos is not None:
            direction = self.get_direction_to_goal(current_pos, dropoff_pos)
            
            # Use some randomization to break loops - every 5 moves, try random
            if self.random_counter % 5 == 0 and len(valid_moves) > 1:
                return random.choice(valid_moves)
            
            # Try direct movement first, but use 4-directional only (environment constraint)
            if direction == "N" and "N" in valid_moves:
                return "N"
            elif direction == "S" and "S" in valid_moves:
                return "S"
            elif direction == "E" and "E" in valid_moves:
                return "E"
            elif direction == "W" and "W" in valid_moves:
                return "W"
            elif direction == "NE":
                # Try N first, then E, but add randomization
                options = [m for m in ["N", "E"] if m in valid_moves]
                if options:
                    return random.choice(options) if self.random_counter % 3 == 0 else options[0]
            elif direction == "NW":
                # Try N first, then W, but add randomization
                options = [m for m in ["N", "W"] if m in valid_moves]
                if options:
                    return random.choice(options) if self.random_counter % 3 == 0 else options[0]
            elif direction == "SE":
                # Try S first, then E, but add randomization
                options = [m for m in ["S", "E"] if m in valid_moves]
                if options:
                    return random.choice(options) if self.random_counter % 3 == 0 else options[0]
            elif direction == "SW":
                # Try S first, then W, but add randomization
                options = [m for m in ["S", "W"] if m in valid_moves]
                if options:
                    return random.choice(options) if self.random_counter % 3 == 0 else options[0]
        
        # Rules 11-18: If not carrying item, move toward pickup based on 8-directional relationship
        if not has_item and pickup_pos is not None:
            direction = self.get_direction_to_goal(current_pos, pickup_pos)
            
            # Use some randomization to break loops - every 5 moves, try random
            if self.random_counter % 5 == 0 and len(valid_moves) > 1:
                return random.choice(valid_moves)
            
            # Try direct movement first, but use 4-directional only (environment constraint)
            if direction == "N" and "N" in valid_moves:
                return "N"
            elif direction == "S" and "S" in valid_moves:
                return "S"
            elif direction == "E" and "E" in valid_moves:
                return "E"
            elif direction == "W" and "W" in valid_moves:
                return "W"
            elif direction == "NE":
                # Try N first, then E, but add randomization
                options = [m for m in ["N", "E"] if m in valid_moves]
                if options:
                    return random.choice(options) if self.random_counter % 3 == 0 else options[0]
            elif direction == "NW":
                # Try N first, then W, but add randomization
                options = [m for m in ["N", "W"] if m in valid_moves]
                if options:
                    return random.choice(options) if self.random_counter % 3 == 0 else options[0]
            elif direction == "SE":
                # Try S first, then E, but add randomization
                options = [m for m in ["S", "E"] if m in valid_moves]
                if options:
                    return random.choice(options) if self.random_counter % 3 == 0 else options[0]
            elif direction == "SW":
                # Try S first, then W, but add randomization
                options = [m for m in ["S", "W"] if m in valid_moves]
                if options:
                    return random.choice(options) if self.random_counter % 3 == 0 else options[0]
        
        # Rule 19: No rule applies -> random valid action
        if valid_moves:
            return random.choice(valid_moves)
        
        # Final fallback if somehow no moves are valid
        return "WAIT"


def run_single_episode(randomize_layout: bool = True, random_start: bool = True, max_steps: int = 50, show_animation: bool = True) -> Dict:
    """
    Single-episode runner for the reflex agent.
    
    Args:
        randomize_layout: Whether to randomize pickup/dropoff positions
        random_start: Whether to start agent at random position
        max_steps: Maximum steps to run before truncation
        show_animation: Whether to display the animation
        
    Returns:
        Dictionary with episode results and data
    """
    print("🤖 Starting Simple Reflex Agent Episode")
    print("=" * 45)
    
    # Initialize environment and agent
    env = WarehouseEnv(max_steps=max_steps)
    agent = SimpleReflexAgent()
    
    # Reset environment
    obs = env.reset(randomize=randomize_layout)
    
    # Move robot to random starting position if requested
    if random_start:
        random_pos = env._random_empty_cell()
        env.state.robot_pos = random_pos
        obs = env._observe()
        print(f"🎲 Random start position: {random_pos}")
    
    print(f"🤖 Robot starting at: {obs['robot_pos']}")
    print(f"📦 Pickup location: {obs['pickup_pos']}")
    print(f"🎯 Dropoff location: {obs['dropoff_pos']}")
    print(f"🔋 Battery: {obs['battery']}")
    
    # Data collection for animation
    frames = []
    metrics = {
        "rewards": [],
        "battery": [],
        "dist_pickup": [],
        "dist_dropoff": []
    }
    
    step_count = 0
    total_reward = 0
    
    print(f"\n🎮 Running episode (max {max_steps} steps)...")
    
    def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    while step_count < max_steps:
        # Capture current frame
        frames.append(env.render_grid())
        
        # Calculate distances for metrics
        current_pos = obs["robot_pos"]
        pickup_pos = obs["pickup_pos"]
        dropoff_pos = obs["dropoff_pos"]
        
        dist_pickup = manhattan_distance(current_pos, pickup_pos) if pickup_pos else 0
        dist_dropoff = manhattan_distance(current_pos, dropoff_pos) if dropoff_pos else 0
        
        # Record metrics for current state (before taking action)
        metrics["battery"].append(obs["battery"])
        metrics["dist_pickup"].append(dist_pickup)
        metrics["dist_dropoff"].append(dist_dropoff)
        
        # Get action from agent
        action = agent.act(obs, env)
        
        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        total_reward += reward
        
        # Record reward after taking action
        metrics["rewards"].append(reward)
        
        # Log step
        status = ""
        if "invalid_action" in info:
            status = " ❌"
        elif reward >= 9.0:
            status = " 🎉 TASK COMPLETE!"
        elif reward >= 4.0:
            status = " 📦 PICKED UP"
        
        print(f"  Step {step_count}: {action} -> {obs['robot_pos']} (reward: {reward:.2f}){status}")
        
        # Check termination conditions
        if terminated:
            print(f"\n✅ Episode completed successfully in {step_count} steps!")
            break
        elif truncated:
            print(f"\n⏰ Episode truncated after {step_count} steps")
            break
    
    # Add final frame and final metrics
    frames.append(env.render_grid())
    
    # Add final metrics to match frame count
    final_pos = obs["robot_pos"]
    final_pickup_pos = obs["pickup_pos"]
    final_dropoff_pos = obs["dropoff_pos"]
    
    final_dist_pickup = manhattan_distance(final_pos, final_pickup_pos) if final_pickup_pos else 0
    final_dist_dropoff = manhattan_distance(final_pos, final_dropoff_pos) if final_dropoff_pos else 0
    
    metrics["battery"].append(obs["battery"])
    metrics["dist_pickup"].append(final_dist_pickup)
    metrics["dist_dropoff"].append(final_dist_dropoff)
    metrics["rewards"].append(0.0)  # No reward for final state
    
    # Episode summary
    print(f"\n📊 Episode Summary:")
    print(f"   Steps taken: {step_count}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Final battery: {obs['battery']}")
    print(f"   Task completed: {'✅ Yes' if terminated else '❌ No'}")
    print(f"   Generated {len(frames)} animation frames")
    
    # Create and display animation
    if show_animation:
        print(f"\n🎬 Creating animation...")
        try:
            anim = replay_animation(
                frames=frames,
                metrics=metrics,
                interval_ms=500,  # Slower for better visibility of reflex decisions
                speed=1.0
            )
            
            if anim:
                print("✅ Animation created successfully!")
                print("   Controls: SPACE=pause/resume, LEFT/RIGHT=navigate frames")
        except Exception as e:
            print(f"❌ Animation error: {e}")
    
    return {
        "frames": frames,
        "metrics": metrics,
        "steps": step_count,
        "total_reward": total_reward,
        "completed": terminated,
        "final_obs": obs,
        "final_battery": obs["battery"],
        "truncated": truncated
    }


def run_multiple_episodes(num_episodes: int = 10, max_steps: int = 100) -> Dict:
    """
    Run multiple episodes to analyze reflex agent performance.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary with aggregated results and analysis
    """
    print(f"\n🔬 Running {num_episodes} Episodes for Reflex Agent Analysis")
    print("=" * 60)
    
    results = []
    success_count = 0
    total_steps = 0
    total_rewards = 0
    failure_modes = {
        "battery_depletion": 0,
        "time_limit": 0,
        "success": 0
    }
    
    for episode in range(num_episodes):
        print(f"\n📋 Episode {episode + 1}/{num_episodes}")
        print("-" * 30)
        
        # Run episode without animation for speed
        result = run_single_episode(
            randomize_layout=True,
            random_start=True,
            max_steps=max_steps,
            show_animation=False
        )
        
        results.append(result)
        total_steps += result["steps"]
        total_rewards += result["total_reward"]
        
        # Analyze failure modes
        if result["completed"]:
            success_count += 1
            failure_modes["success"] += 1
        elif result["truncated"]:
            if result["final_battery"] <= 0:
                failure_modes["battery_depletion"] += 1
                print("   ⚡ Failure: Battery depleted")
            else:
                failure_modes["time_limit"] += 1
                print("   ⏰ Failure: Time limit reached")
    
    # Summary statistics
    avg_steps = total_steps / num_episodes
    avg_reward = total_rewards / num_episodes
    success_rate = success_count / num_episodes * 100
    
    print(f"\n📊 Multi-Episode Analysis Results:")
    print(f"   Success rate: {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"   Average steps: {avg_steps:.1f}")
    print(f"   Average reward: {avg_reward:.2f}")
    print(f"\n🔍 Failure Mode Analysis:")
    print(f"   ✅ Successful completions: {failure_modes['success']}")
    print(f"   ⚡ Battery depletion: {failure_modes['battery_depletion']}")
    print(f"   ⏰ Time limit exceeded: {failure_modes['time_limit']}")
    
    return {
        "results": results,
        "summary": {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "failure_modes": failure_modes
        }
    }


def demo_reflex_rules():
    """
    Demonstrate the reflex rules in action with a simple scenario.
    """
    print("\n🔍 Demonstrating Reflex Agent Rules")
    print("=" * 40)
    
    # Create a simple environment for demonstration
    env = WarehouseEnv(max_steps=20)
    agent = SimpleReflexAgent()
    
    obs = env.reset(randomize=True)
    
    print(f"🤖 Robot at: {obs['robot_pos']}")
    print(f"📦 Pickup at: {obs['pickup_pos']}")
    print(f"🎯 Dropoff at: {obs['dropoff_pos']}")
    print(f"🎒 Carrying item: {obs['has_item']}")
    print()
    
    # Show what action the agent would take and why
    current_pos = obs["robot_pos"]
    has_item = obs["has_item"]
    pickup_pos = obs["pickup_pos"]
    dropoff_pos = obs["dropoff_pos"]
    
    print("🧠 Reflex Rule Analysis:")
    
    # Check each rule in order
    if not has_item and pickup_pos == current_pos:
        print("   Rule 1 MATCHES: At pickup and no item → PICK")
        action = "PICK"
    elif has_item and dropoff_pos == current_pos:
        print("   Rule 2 MATCHES: At dropoff and carrying item → DROP")
        action = "DROP"
    elif has_item and dropoff_pos is not None:
        direction = agent.get_direction_to_goal(current_pos, dropoff_pos)
        print(f"   Rule 3 MATCHES: Carrying item → dropoff is {direction}")
        action = direction if direction in ["N", "S", "E", "W"] else "N"  # Fallback for diagonal
    elif not has_item and pickup_pos is not None:
        direction = agent.get_direction_to_goal(current_pos, pickup_pos)
        print(f"   Rule 4 MATCHES: Not carrying → pickup is {direction}")
        action = direction if direction in ["N", "S", "E", "W"] else "N"  # Fallback for diagonal
    else:
        print("   Rule 5 MATCHES: No specific rule → random valid move")
        valid_moves = agent.get_valid_moves(env, current_pos)
        action = random.choice(valid_moves) if valid_moves else "WAIT"
    
    print(f"\n🎬 Selected Action: {action}")
    
    return {"demo_action": action, "obs": obs}


if __name__ == "__main__":
    """
    Main execution: Test the simple reflex agent.
    """
    print("🤖 Simple Reflex Agent for Warehouse Environment")
    print("=" * 55)
    
    # Demonstrate the reflex rules
    demo_result = demo_reflex_rules()
    
    # Run single episode with animation
    print("\n" + "="*60)
    print("🎮 Single Episode Demo")
    result = run_single_episode(
        randomize_layout=True,
        random_start=True,
        max_steps=75,
        show_animation=True
    )
    
    print(f"\n🏁 Demo Episode Results:")
    print(f"   Result: {'SUCCESS' if result['completed'] else 'INCOMPLETE'}")
    print(f"   Performance: {result['total_reward']:.2f} reward in {result['steps']} steps")
    print(f"   Final battery: {result['final_battery']}")
    
    # Run multiple episodes for analysis
    print(f"\n" + "="*60)
    analysis = run_multiple_episodes(num_episodes=15, max_steps=100)
    
    # Final analysis and comparison notes
    success_rate = analysis["summary"]["success_rate"]
    print(f"\n🔧 Reflex Agent Performance Analysis:")
    
    if success_rate >= 70:
        print("   ✅ Reflex agent performs reasonably well")
        print("   💡 Simple rules can be effective for basic navigation")
    elif success_rate >= 40:
        print("   ⚠️ Moderate performance - reflex rules have limitations")
        print("   💡 May struggle with obstacles or complex layouts")
    else:
        print("   ❌ Poor performance - simple reflex approach insufficient")
        print("   💡 Consider: Memory-based agents or more sophisticated planning")
    
    print(f"\n📝 Key Insights:")
    print(f"   • Reflex agents are stateless - no memory of past actions")
    print(f"   • Performance depends heavily on environment complexity")
    print(f"   • Simple rules work well for straightforward scenarios")
    print(f"   • May get stuck in loops or suboptimal paths without memory")
    
    avg_steps = analysis["summary"]["avg_steps"]
    print(f"\n📈 Efficiency Metrics:")
    print(f"   • Average completion time: {avg_steps:.1f} steps")
    print(f"   • Success rate: {success_rate:.1f}%")
    print(f"   • Rule-based decision making: 100% deterministic (except random fallback)")
