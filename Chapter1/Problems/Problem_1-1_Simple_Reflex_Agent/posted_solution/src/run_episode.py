from warehouse_env import WarehouseEnv

def run_episode(env, agent, max_steps=100, record_frames=False):
    """Run a single episode with the given agent in the environment."""
    obs = env.reset()  # reset() returns the initial observation
    total_reward = 0
    steps = 0
    
    while steps < max_steps:
        # Agent chooses action - check if agent uses act() with env parameter or just state
        if hasattr(agent, 'act'):
            # Try both interfaces - some agents take (obs, env) or just (obs)
            try:
                action = agent.act(obs, env)
            except TypeError:
                # If that fails, try just with obs
                action = agent.act(obs)
        else:
            # Fallback to get_action if act method doesn't exist
            action = agent.get_action(obs)
        
        # Execute action in environment
        # step() returns (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    return {
        'terminated': terminated,
        'steps': steps,
        'battery': getattr(agent, 'battery', obs.get('battery', None)),  # Safe way to get battery
        'total_reward': total_reward
    }

def run_agent(agent_class, label, num_episodes=50):
    """Run multiple episodes with an agent class and collect statistics."""
    env = WarehouseEnv()
    stats = {
        'success': [],
        'steps': [],
        'battery': [],
        'reward': [],
    }
    for _ in range(num_episodes):
        agent = agent_class()
        result = run_episode(env, agent, record_frames=False)
        stats['success'].append(1 if result['terminated'] else 0)
        stats['steps'].append(result['steps'])
        stats['battery'].append(result['battery'] if result['battery'] is not None else 0)
        stats['reward'].append(result['total_reward'])
    return stats