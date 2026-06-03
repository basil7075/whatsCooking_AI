import os
import shutil
import webbrowser
import threading
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ai import load_pdf, find_dishes, get_recipe

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("index.html")

@app.on_event("startup")
async def startup_event():
    try:
        url = "http://127.0.0.1:8000"
        def _open():
            time.sleep(0.5)
            webbrowser.open_new_tab(url)
        threading.Thread(target=_open, daemon=True).start()
        print(f"Frontend available at {url}")
    except Exception as e:
        print("Could not open browser:", e)

class IngredientsRequest(BaseModel):
    ingredients: str

class RecipeRequest(BaseModel):
    dish_name: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        message = load_pdf(temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(temp_path)
    
    return {"message": message}


@app.post("/dishes")
def dishes(request: IngredientsRequest):
    try:
        result = find_dishes(request.ingredients)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    response_data = {"dishes": result}
    print(f"[DEBUG] Final API response for /dishes: {response_data}")
    return response_data


@app.post("/recipe")
def recipe(request: RecipeRequest):
    try:
        result = get_recipe(request.dish_name)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"recipe": result}