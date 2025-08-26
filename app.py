# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import requests
import tempfile
import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
from preprocess import preprocess_data # Importe a nova função

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

# Carregar modelo
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
    d = data.model_dump()

    # Crie um DataFrame a partir dos dados do usuário
    df_raw = pd.DataFrame([d])

    # Pré-processamento
    df_processed = preprocess_data(df_raw, le_dict=le_dict, is_training=False)
    
    # Reordenar as colunas para coincidir com as colunas do modelo
    df_final = df_processed.reindex(columns=feature_columns, fill_value=0)

    # Predição
    pred_encoded = model.predict(df_final)
    cargo_previsto = le_dict["cargo"].inverse_transform(pred_encoded)
    cargo_previsto_str = str(cargo_previsto[0])

    return {
        "input": d,
        "cargo_previsto": cargo_previsto_str
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)