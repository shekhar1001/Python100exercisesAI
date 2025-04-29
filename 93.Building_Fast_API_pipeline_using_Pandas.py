from fastapi import FastAPI, UploadFile, File
import pandas as pd
from io import StringIO

app = FastAPI()

@app.post("/clean")
async def clean_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode()))
    
    # Example cleaning: drop NA
    df_clean = df.dropna()

    return {
        "rows_before": len(df),
        "rows_after": len(df_clean),
        "columns": df_clean.columns.tolist()
    }
