from fastapi import FastAPI, UploadFile, File
from audio_processing import predict_audio

app = FastAPI(debug=True)

@app.post("/predict")
async def predict(audio_file: UploadFile = File(...)):
    try:
        predicted_class = predict_audio(audio_file.file)
        return {"predicted_class": predicted_class} 
    except Exception as e:
        return {"error": str(e)}
