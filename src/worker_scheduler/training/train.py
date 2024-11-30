from ..environment.env import WorkerSchedulingEnv
from ..agents.ppo import PPO

def run_training(env , render_freq=200, save_dir='./models'):
    """Utility function to create environment and start training"""
    
    ppo = PPO(env)
    metrics = ppo.train(render_freq=render_freq, save_dir=save_dir)
    
    return ppo, metrics

def evaluate_model(model_path, env, num_episodes=10):
    """Utility function to evaluate a trained model"""
    ppo = PPO(env)
    episode = ppo.load_model(model_path)
    print(f"Loaded model from episode {episode}")
    
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _, _ = ppo.get_action_and_value(state, training=False)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
        
        print(f"Episode {ep+1}, Total Reward: {total_reward:.2f}")
