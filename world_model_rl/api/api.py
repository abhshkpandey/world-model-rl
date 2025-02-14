from fastapi import FastAPI, HTTPException
import torch
from world_model_rl import WorldModelRL

app = FastAPI()

# Load the model
model = WorldModelRL()
model.eval()

@app.get("/")
def root():
    return {"message": "World Model RL API is running"}

@app.post("/predict")
def predict(state: list):
    """
    Get an action prediction from the model.
    :param state: A list representing the state input.
    """
    try:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = model(state_tensor)
        return {"action": action[1].tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train")
def train():
    """
    Trigger model training.
    """
    try:
        from core.trainer import Trainer
        from core.config import CONFIG
        trainer = Trainer(model, CONFIG)
        trainer.train()
        return {"message": "Training started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
