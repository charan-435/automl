from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
import shutil
import os
from model import train_model, predict_model
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

global_df = None
global_model = None
global_columns = None
@app.get("/datasets")
async def get_datasets():
    files = os.listdir("data/datasets")
    return {"datasets": files}

@app.post("/load_dataset")
async def load_dataset(name: str = Form(...)):
    global global_df
    path = f"data/datasets/{name}"
    if not os.path.exists(path):
        return {"error": "Dataset not found"}
    global_df = pd.read_csv(path)
    return {"columns": list(global_df.columns)}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global global_df

    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    global_df = pd.read_csv(file_path)

    return {
        "columns": list(global_df.columns)
    }


@app.post("/train")
async def train(target: str ):
    global global_df, global_model, global_columns

    if global_df is None:
        return {"error": "No dataset uploaded"}

    model, scores, columns = train_model(global_df, target)

    global_model = model
    global_columns = columns

    return {
        "scores": scores
    }


@app.get("/eda")
async def eda():
    global global_df, global_model, global_columns

    if global_df is None:
        return {"error": "No dataset"}

    import numpy as np
    df = global_df.copy()

    # -------------------------------
    # Summary stats
    # -------------------------------
    summary = df.describe(include="all").fillna("").to_dict()

    # -------------------------------
    # Correlation
    # -------------------------------
    corr = df.select_dtypes(include=["number"]).corr().fillna(0).to_dict()

    # -------------------------------
    # Histograms (REAL bins)
    # -------------------------------
    histograms = {}
    for col in df.select_dtypes(include=["number"]).columns:
        counts, bins = np.histogram(df[col], bins=10)

        histograms[col] = [
            {
                "bin": f"{round(bins[i],2)} - {round(bins[i+1],2)}",
                "count": int(counts[i])
            }
            for i in range(len(counts))
        ]

    # -------------------------------
    # Categorical (Pie charts)
    # -------------------------------
    categorical = {}
    for col in df.select_dtypes(include=["object"]).columns:
        categorical[col] = df[col].value_counts().head(10).to_dict()

    # -------------------------------
    # Missing values
    # -------------------------------
    missing = df.isnull().sum().to_dict()

    # -------------------------------
    # Feature importance (if model trained)
    # -------------------------------
    importance = {}
    if global_model is not None and hasattr(global_model, "feature_importances_"):
        importance = dict(zip(global_columns, global_model.feature_importances_))

    return {
        "columns": list(df.columns),
        "sample": df.head(5).to_dict(orient="records"),
        "summary": summary,
        "correlation": corr,
        "histograms": histograms,
        "categorical": categorical,
        "missing": missing,
        "importance": importance
    }

@app.post("/predict")
async def predict(data: dict):
    try:
       prediction = predict_model(global_model, data, global_columns)
       return {"prediction": prediction}
    except Exception as e:
         return {"error": str(e)}
    