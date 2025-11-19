# calctwo.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from calcone import list_image_files, train_and_save_model, predict_for_files, MODEL_PATH
import os

app = FastAPI(title="Autoencoder API")

class TrainRequest(BaseModel):
    latent_dim: Optional[int] = 164
    epochs: Optional[int] = 10
    batch_size: Optional[int] = 16
    validation_split: Optional[float] = 0.2

class PredictRequest(BaseModel):
    filenames: List[str]

@app.get("/list_images/")
def api_list_images():
    files = list_image_files()
    return {"images": files}

@app.post("/train/")
def api_train(req: TrainRequest):
    # Train model and save
    try:
        history = train_and_save_model(latent_dim=req.latent_dim,
                                       epochs=req.epochs,
                                       batch_size=req.batch_size,
                                       validation_split=req.validation_split)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "trained", "model_path": MODEL_PATH, "history": history}

@app.post("/predict/")
def api_predict(req: PredictRequest):
    try:
        results = predict_for_files(req.filenames)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"results": results}

# For local dev: start with:
# uvicorn calctwo:app --reload
if __name__ == "__main__":
    # uvicorn.run("calctwo:app", host="127.0.0.1", port=8000, reload=True)
    print("Run this FastAPI app on a local machine or a cloud server, not on Streamlit Cloud")

