from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins for CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint to read data from output.txt
@app.get("/read_data")
def read_data():
    try:
        with open("output.txt", "r") as file:
            data = file.read()
        return {"data": data}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

# Endpoint to append data to output.txt
@app.post("/append_data")
def append_data(data: str):
    try:
        with open("output.txt", "a") as file:
            file.write(data + "\n")
        return {"status": "Data appended successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Additional endpoint to upload a file and append its content to output.txt
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open("output.txt", "a") as file:
            file.write(contents.decode("utf-8") + "\n")
        return {"status": "File content appended successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to download the output.txt file
@app.get("/download_file")
def download_file():
    return FileResponse("output.txt", filename="output.txt")


