import os, sys
from stable_baselines3 import MaskablePPO
# Import the Hardware Environment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.decoder_env_xsim import DecoderEnvXSim
# from sb3_contrib import MaskablePPO # If using sb3-contrib

def run_hardware_inference(image_folder, sim_dir, model_path, output_hex_path="run_history.hex", max_cycles=50000):
    """
    Runs a single inference episode on the Xsim environment.
    
    Args:
        image_folder: Path to the specific image stem to load
        model_path: Path to the trained .zip model
        output_hex_path: Where to save the raw hardware actions
        max_cycles: Deadlock threshold
    """
    # Initialize environment specifically for this image
    env = DecoderEnvXSim(data_dir=image_folder, sim_dir=sim_dir, is_inference=True)
    model = MaskablePPO.load(model_path)
    
    obs, info = env.reset()
    cycle_count = 0
    deadlock_detected = False
    
    # Open the log file to store the historical actions
    with open(output_hex_path, "w") as hex_log:
        
        while True:
            # deterministic=True disables random exploration noise for pure inference
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the hardware simulation
            obs, reward, terminated, truncated, info = env.step(action)
            cycle_count += 1
            
            # Capture the raw hex dispatched to the hardware
            # Since env.step() just overwrote action.txt, we read it back immediately
            if os.path.exists("action.txt"):
                with open("action.txt", "r") as f:
                    last_action = f.read().strip()
                    if last_action:
                        hex_log.write(f"{last_action}\n")
            
            # Check completion based on your FIFO/Command exhaustion logic
            if terminated or truncated:
                break
                
            # Deadlock watchdog
            if cycle_count >= max_cycles:
                deadlock_detected = True
                break

    # Force cleanup to prevent zombie xsimk processes
    env.close()
    
    if deadlock_detected:
        print(f"❌ DEADLOCK DETECTED: Simulation forcefully halted at {cycle_count} cycles.")
    else:
        print(f"✅ INFERENCE COMPLETE: Image successfully decoded in {cycle_count} cycles.")
        
    return cycle_count, deadlock_detected

# Example Execution:
# cycles, deadlocked = run_hardware_inference(FpgaEnv, "./images/test_image_01", "./models/best_agent.zip")