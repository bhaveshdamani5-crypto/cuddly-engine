from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
import random

try:
    from stable_baselines3 import PPO
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False

app = FastAPI(
    title="SafeGuard-OpenEnv: Industrial RL AI Controller",
    description="Meta Hackathon Gym RL Server",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RL Agent variable
rl_agent = None

class SensorData(BaseModel):
    temperature: float
    humidity: float
    co2: float
    pm25: float
    
class RLResponse(BaseModel):
    action_id: int
    action_name: str
    llama_assessment: str

# Meta Llama 3 generation engine (mock for hackathon speed)
def generate_llama_report(state, action):
    if action == 1:
        return f"Llama-3 Analysis: PM2.5 or CO2 levels approaching danger thresholds (CO2: {state[2]:.0f}ppm). Engaging emergency ventilation to safely purge the toxic accumulation."
    elif action == 2:
        return f"Llama-3 Analysis: Sensor suite detected critical thermal buildup (Temp: {state[0]:.1f}C). Activating high-power cooling loop to prevent hardware meltdown."
    elif action == 3:
        return f"Llama-3 Analysis: CATASTROPHIC DIVERGENCE DETECTED. Environment is fully hostile. Requesting absolute Emergency Facility Shutdown to avoid disaster."
    else:
        return "Llama-3 Analysis: All telemetry streams report baseline operations. Energy optimization protocol enabled (No mitigation systems active)."

@app.on_event("startup")
async def load_rl_agent():
    global rl_agent
    try:
        if STABLE_BASELINES_AVAILABLE and os.path.exists('models/ppo_safeguard.zip'):
            rl_agent = PPO.load('models/ppo_safeguard.zip')
            print("Successfully loaded PPO RL Agent.")
        else:
            print("Warning: RL Agent not found or stable-baselines3 missing. Using fallback policy for frontend demo demo.")
    except Exception as e:
        print(f"Error loading RL Agent: {e}")

@app.post("/predict_action", response_model=RLResponse)
async def predict_optimal_action(data: SensorData):
    state_vector = np.array([data.temperature, data.humidity, data.co2, data.pm25], dtype=np.float32)

    try:
        # Action Map based on env/safeguard_env.py
        actions = {
            0: "CONTINUE_OPERATIONS",
            1: "TRIGGER_VENTILATION",
            2: "TRIGGER_COOLING",
            3: "EMERGENCY_SHUTDOWN"
        }

        # Inference from the trained Gym Agent
        if rl_agent:
            action, _states = rl_agent.predict(state_vector, deterministic=True)
            action_id = int(action)
        else:
            # Fallback algorithmic heuristic if no RL agent trained locally
            action_id = 0
            if data.temperature > 50.0 or data.co2 > 1500 or data.pm25 > 200:
                action_id = 3
            elif data.co2 > 800 or data.pm25 > 50:
                action_id = 1
            elif data.temperature > 35:
                action_id = 2

        action_name = actions.get(action_id, "CONTINUE_OPERATIONS")
        report = generate_llama_report(state_vector, action_id)

        return {
            "action_id": action_id,
            "action_name": action_name,
            "llama_assessment": report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "rl_agent_loaded": rl_agent is not None}
