"""
Greedy Manhattan Agent with Loop Detection for Warehouse Environment
==================================================================

This agent implements:
1. Greedy Manhattan distance movement toward current goal
2. Loop detection using last 10 positions
3. Random escape moves for 3 steps when stuck in a loop
4. Fallback to random valid moves when stuck

Author: Created for EAI Chapter 1
"""

import random
from typing import List, Tuple, Dict, Optional
from collections import deque

from warehouse_env import WarehouseEnv
from warehouse_viz import replay_animation


class GreedyManhattanAgent:
    """
    Agent that uses Manhattan distance to greedily move toward goals.
    Features loop detection and random escape behavior.
    
    Core Algorithm:
    1. Compute Manhattan distance to current goal (pickup if no item, dropoff if carrying)
    2. Choose action (N/S/E/W) that reduces distance most
    3. Fall back to random valid move if stuck (all moves increase distance)
    4. Loop detector: track last N=10 positions; trigger random escape if position repeated
    """
    
    def __init__(self):
        self.position_history = deque(maxlen=12)  # Track last 12 positions for better loop detection
        self.escape_mode = False
        self.escape_steps_remaining = 0
        self.escape_duration = 6  # Increased escape duration
        self.loop_detections = 0  # Count loop detections for analysis
        self.failed_moves = set()  # Track recently failed moves to avoid repeating them
        
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
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
    
    def detect_loop(self, current_pos: Tuple[int, int]) -> bool:
        """
        Detect if the agent is stuck in a loop.
        Returns True if the current position appears frequently in recent history.
        """
        if len(self.position_history) < 6:  # Need some history to detect loops
            return False
        
        # Count occurrences of current position in recent history
        position_count = sum(1 for pos in self.position_history if pos == current_pos)
        
        # If we've been at this position more than 2 times in last 10 steps, it's likely a loop
        return position_count >= 3
    
    def get_greedy_action(self, obs: Dict, env: WarehouseEnv) -> str:
        """
        Get the greedy action based on Manhattan distance to current goal.
        """
        current_pos = obs["robot_pos"]
        has_item = obs["has_item"]
        
        # Determine current goal
        if has_item:
            goal = obs["dropoff_pos"]
            goal_name = "dropoff"
        else:
            goal = obs["pickup_pos"]
            goal_name = "pickup"
        
        if goal is None:
            return "WAIT"  # No valid goal
        
        # Get valid moves
        valid_moves = self.get_valid_moves(env, current_pos)
        
        if not valid_moves:
            return "WAIT"  # Stuck with no valid moves
        
        # Find the move that gets us closest to the goal while avoiding recent failures
        best_action = None
        best_distance = float('inf')
        recent_positions = set(list(self.position_history)[-5:])  # Avoid recent positions
        
        # First pass: try moves that don't revisit recent positions
        good_moves = []
        for action in valid_moves:
            dr, dc = env.MOVE_DELTAS[action]
            r, c = current_pos
            new_pos = (r + dr, c + dc)
            distance = self.manhattan_distance(new_pos, goal)
            
            # Skip moves to recently visited positions or failed moves
            if new_pos in recent_positions or action in self.failed_moves:
                continue
                
            good_moves.append((action, distance))
            if distance < best_distance:
                best_distance = distance
                best_action = action
        
        # If we found good moves, use the best one
        if best_action:
            return best_action
            
        # Fallback: if all moves lead to recent positions, clear failed moves and try again
        self.failed_moves.clear()
        for action in valid_moves:
            dr, dc = env.MOVE_DELTAS[action]
            r, c = current_pos
            new_pos = (r + dr, c + dc)
            distance = self.manhattan_distance(new_pos, goal)
            
            if distance < best_distance:
                best_distance = distance
                best_action = action
        
        return best_action if best_action else random.choice(valid_moves)
    
    def act(self, obs: Dict, env: WarehouseEnv) -> str:
        """
        Main action selection method with loop detection and escape behavior.
        """
        current_pos = obs["robot_pos"]
        has_item = obs["has_item"]
        
        # Update position history
        self.position_history.append(current_pos)
        
        # Check if we're at pickup/dropoff location and should take action
        if not has_item and obs["pickup_pos"] == current_pos:
            return "PICK"
        elif has_item and obs["dropoff_pos"] == current_pos:
            return "DROP"
        
        # Handle escape mode
        if self.escape_mode:
            self.escape_steps_remaining -= 1
            if self.escape_steps_remaining <= 0:
                self.escape_mode = False
                self.failed_moves.clear()  # Reset failed moves after escape
                print(f"  🔄 Exiting escape mode at {current_pos}")
            else:
                # Make smart escape move - avoid recently visited positions
                valid_moves = self.get_valid_moves(env, current_pos)
                if valid_moves:
                    # Prefer moves that take us to positions we haven't been recently
                    recent_positions = set(list(self.position_history)[-8:])
                    escape_moves = []
                    
                    for action in valid_moves:
                        dr, dc = env.MOVE_DELTAS[action]
                        r, c = current_pos
                        new_pos = (r + dr, c + dc)
                        if new_pos not in recent_positions:
                            escape_moves.append(action)
                    
                    # If we have moves to unvisited positions, use those
                    action = random.choice(escape_moves) if escape_moves else random.choice(valid_moves)
                    print(f"  🎲 Escape move {self.escape_duration - self.escape_steps_remaining}: {action}")
                    return action
                else:
                    return "WAIT"
        
        # Detect loop
        if self.detect_loop(current_pos):
            if not self.escape_mode:  # Only trigger if not already in escape mode
                self.escape_mode = True
                self.escape_steps_remaining = self.escape_duration
                self.loop_detections += 1
                print(f"  🚨 Loop detected at {current_pos}! Entering escape mode for {self.escape_duration} steps (Detection #{self.loop_detections})")
                # Clear some history to avoid immediate re-detection
                for _ in range(4):
                    if self.position_history:
                        self.position_history.popleft()
                # Return immediately to start escape
                return self.act(obs, env)
        
        # Normal greedy behavior
        action = self.get_greedy_action(obs, env)
        
        # If greedy action is None or WAIT, try smarter fallback
        if action in [None, "WAIT"]:
            valid_moves = self.get_valid_moves(env, current_pos)
            if valid_moves:
                # Track this as a potential failure point
                recent_positions = set(list(self.position_history)[-3:])
                fallback_moves = [move for move in valid_moves 
                                 if move not in self.failed_moves]
                
                if fallback_moves:
                    action = random.choice(fallback_moves)
                else:
                    # All moves have failed recently, clear failed moves and try again
                    self.failed_moves.clear()
                    action = random.choice(valid_moves)
                    
                print(f"  🎯 Fallback move: {action}")
        
        # Track if this move doesn't make progress (for next iteration)
        if action and action != "WAIT":
            dr, dc = env.MOVE_DELTAS[action]
            r, c = current_pos
            new_pos = (r + dr, c + dc)
            
            # If this move takes us to a recently visited position, mark it as potentially failed
            recent_positions = set(list(self.position_history)[-3:])
            if new_pos in recent_positions:
                self.failed_moves.add(action)
        
        return action


def run_single_episode(randomize_layout: bool = False, random_start: bool = True, max_steps: int = 50, show_animation: bool = True) -> Dict:
    """
    Single-episode runner that resets environment, runs episode, and replays animation.
    
    Args:
        randomize_layout: Whether to randomize pickup/dropoff positions
        random_start: Whether to start agent at random position
        max_steps: Maximum steps to run before truncation
        show_animation: Whether to display the animation
        
    Returns:
        Dictionary with episode results and data
    """
    print("🏭 Starting Warehouse Agent Episode")
    print("=" * 40)
    
    # Initialize environment and agent
    env = WarehouseEnv(max_steps=max_steps)
    agent = GreedyManhattanAgent()
    
    # Reset environment with random start position
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
    
    while step_count < max_steps:
        # Capture current frame
        frames.append(env.render_grid())
        
        # Calculate distances for metrics
        current_pos = obs["robot_pos"]
        pickup_pos = obs["pickup_pos"]
        dropoff_pos = obs["dropoff_pos"]
        
        dist_pickup = agent.manhattan_distance(current_pos, pickup_pos) if pickup_pos else 0
        dist_dropoff = agent.manhattan_distance(current_pos, dropoff_pos) if dropoff_pos else 0
        
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
    
    final_dist_pickup = agent.manhattan_distance(final_pos, final_pickup_pos) if final_pickup_pos else 0
    final_dist_dropoff = agent.manhattan_distance(final_pos, final_dropoff_pos) if final_dropoff_pos else 0
    
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
                interval_ms=400,  # Slower for better visibility
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
        "escape_activations": agent.loop_detections,
        "truncated": truncated
    }


def run_multiple_episodes(num_episodes: int = 5, max_steps: int = 50) -> Dict:
    """
    Run multiple episodes to test and analyze agent performance.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary with aggregated results and analysis
    """
    print(f"\n🔬 Running {num_episodes} Episodes for Analysis")
    print("=" * 50)
    
    results = []
    success_count = 0
    total_steps = 0
    total_rewards = 0
    failure_modes = {
        "battery_depletion": 0,
        "time_limit": 0,
        "oscillation": 0,
        "success": 0
    }
    
    for episode in range(num_episodes):
        print(f"\n📋 Episode {episode + 1}/{num_episodes}")
        print("-" * 25)
        
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
        
        # Check for potential oscillation (low progress despite many steps)
        efficiency = result["total_reward"] / result["steps"] if result["steps"] > 0 else 0
        if result["steps"] > 20 and efficiency < -0.05:  # Mostly negative rewards
            failure_modes["oscillation"] += 1
            print("   🔄 Potential oscillation detected")
    
    # Summary statistics
    avg_steps = total_steps / num_episodes
    avg_reward = total_rewards / num_episodes
    success_rate = success_count / num_episodes * 100
    
    print(f"\n📊 Multi-Episode Analysis Results:")
    print(f"   Success rate: {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"   Average steps: {avg_steps:.1f}")
    print(f"   Average reward: {avg_reward:.2f}")
    print(f"   \n🔍 Failure Mode Analysis:")
    print(f"   ✅ Successful completions: {failure_modes['success']}")
    print(f"   ⚡ Battery depletion: {failure_modes['battery_depletion']}")
    print(f"   ⏰ Time limit exceeded: {failure_modes['time_limit']}")
    print(f"   🔄 Potential oscillation: {failure_modes['oscillation']}")
    
    return {
        "results": results,
        "summary": {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "failure_modes": failure_modes
        }
    }


def test_loop_detection():
    """
    Test the loop detection by creating a scenario where the agent might get stuck.
    """
    print("\n🔍 Testing Loop Detection Feature")
    print("=" * 35)
    
    # Create a more complex layout that could cause loops
    complex_grid = [
        "############",
        "#..........#",
        "#..####....#",
        "#..#..#....#",
        "#..#..#....#",
        "#..#P.#.D..#",
        "#..####....#",
        "#..........#",
        "############",
    ]
    
    env = WarehouseEnv(grid=complex_grid, start_pos=(1, 1), max_steps=30)
    agent = GreedyManhattanAgent()
    obs = env.reset(randomize=True)
    
    print(f"🤖 Robot starts at: {obs['robot_pos']}")
    print(f"📦 Pickup at: {obs['pickup_pos']}")
    print(f"🎯 Dropoff at: {obs['dropoff_pos']}")
    
    frames = []
    step_count = 0
    
    while step_count < 30:
        frames.append(env.render_grid())
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        if terminated or truncated:
            break
    
    print(f"\n📊 Test completed in {step_count} steps")
    return {"frames": frames, "steps": step_count}


if __name__ == "__main__":
    """
    Main execution: Test and refine the greedy agent.
    """
    print("🏗️ Greedy Manhattan Agent with Loop Detection")
    print("=" * 50)
    
    # Run single episode with animation
    print("\n🎮 Single Episode Demo")
    result = run_single_episode(
        randomize_layout=True, 
        random_start=True,
        max_steps=50,
        show_animation=True
    )
    
    print(f"\n🏁 Demo Episode Results:")
    print(f"   Result: {'SUCCESS' if result['completed'] else 'INCOMPLETE'}")
    print(f"   Performance: {result['total_reward']:.2f} reward in {result['steps']} steps")
    print(f"   Final battery: {result['final_battery']}")
    print(f"   Loop detections: {result['escape_activations']}")
    
    # Run multiple episodes for analysis
    print(f"\n" + "="*60)
    analysis = run_multiple_episodes(num_episodes=10, max_steps=75)
    
    # Test loop detection with complex scenario
    print(f"\n" + "="*60)
    test_result = test_loop_detection()
    
    # Final recommendations
    success_rate = analysis["summary"]["success_rate"]
    print(f"\n🔧 Agent Performance Analysis:")
    if success_rate >= 80:
        print("   ✅ Agent performs well - ready for deployment")
    elif success_rate >= 60:
        print("   ⚠️ Agent shows moderate performance - consider tuning")
        print("   💡 Suggestions: Adjust loop detection sensitivity, increase escape duration")
    else:
        print("   ❌ Agent needs improvement - major issues detected")
        print("   💡 Suggestions: Review greedy algorithm, improve obstacle avoidance")
    
    failure_modes = analysis["summary"]["failure_modes"]
    if failure_modes["battery_depletion"] > 0:
        print("   🔋 Consider: Energy-efficient path planning")
    if failure_modes["oscillation"] > 0:
        print("   🔄 Consider: Stronger loop detection or memory-based navigation")
    if failure_modes["time_limit"] > 0:
        print("   ⏰ Consider: More aggressive goal-seeking behavior")
