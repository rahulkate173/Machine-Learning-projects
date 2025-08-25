from fastapi import FastAPI , Form ,Response ,Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import joblib
from pydantic import BaseModel

## import model and pipeline
scaling = joblib.load('transformationPipeline.joblib')
model = joblib.load('logisticRegression.joblib')

## creating app
app = FastAPI()

## mounting static and templates
app.mount('/static',StaticFiles(directory='static'),name='static')
templates = Jinja2Templates(directory='templates')

## Creating routes 
@app.get('/',response_class=HTMLResponse)
async def home(request : Request):
    return templates.TemplateResponse('index.html',{'request':request})

from fastapi import Form

@app.post("/predict")
async def predict(request: Request,
                  pregnancies: int = Form(...),
                  glucose: int = Form(...),
                  blood_pressure: int = Form(...),
                  skin_thickness: int = Form(...),
                  insulin: int = Form(...),
                  bmi: float = Form(...),
                  dpf: float = Form(...),
                  age: int = Form(...)):

    try:
        # ✅ Step 1: Collect input as a dataframe row
        input_data = [[pregnancies, glucose, blood_pressure,
                       skin_thickness, insulin, bmi, dpf, age]]

        # ✅ Step 2: Apply preprocessing pipeline
        transformed = scaling.transform(input_data)

        # ✅ Step 3: Predict
        prediction = model.predict(transformed)[0]

        # ✅ Step 4: Render result page
        return templates.TemplateResponse(
            "predict.html",
            {"request": request, "prediction": int(prediction)}
        )

    except Exception as e:
        # Error handling with validation
        return templates.TemplateResponse(
            "predict.html",
            {"request": request, "prediction": None, "error": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run(app,host='127.0.0.1',port=8001)