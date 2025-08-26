from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv
from preprocess import preprocess_data, normalize_text
import requests
import tempfile

load_dotenv()
app = FastAPI(title="Previsão de Cargos")

# CORS
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URL = os.getenv("MODEL_URL")
resp = requests.get(MODEL_URL)
resp.raise_for_status()
tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
with open(tmp_file.name, "wb") as f:
    f.write(resp.content)

saved = joblib.load(tmp_file.name)
model = saved["model"]
le_dict = saved["le_dict"]
feature_columns = saved["feature_columns"]

def normalize_user_input(user_data: dict):
    return {k: normalize_text(v) if isinstance(v, str) else v for k, v in user_data.items()}

# Rota de health check
@app.get("/health")
def health():
    return {"status": "ok"}

# Input esperado
class InputData(BaseModel):
    idade: int
    genero: str
    etnia: str
    pcd: str
    vive_no_brasil: str
    estado_moradia: str
    nivel_ensino: str
    formacao: str
    tempo_experiencia_dados: str
    linguagens_preferidas: str
    bancos_de_dados: str
    cloud_preferida: str

@app.post("/predict")
def predict(data: InputData):
    d = normalize_user_input(data.model_dump())
    df_raw = pd.DataFrame([d])
    df_processed = preprocess_data(df_raw, le_dict=le_dict, is_training=False)
    df_final = df_processed.reindex(columns=feature_columns, fill_value=0)

    pred_encoded = model.predict(df_final)
    cargo_previsto = le_dict["cargo"].inverse_transform(pred_encoded)[0]

    return {
        "input": d,
        "cargo_previsto": cargo_previsto
    }
