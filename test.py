import cv2
import face_recognition
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import pickle
import numpy as np
import io
import requests

app = FastAPI()

# URL of the EncodeFile.pkl on GitHub
github_encode_file_url = "https://github.com/m-akrm/vision_project/blob/main/EncodeFile.pkl"
# Load the trained data
try:
    response = requests.get(github_encode_file_url)
    response.raise_for_status()
    encodeKnown = pickle.loads(response.content)
    trained_people, names = encodeKnown
except requests.exceptions.RequestException as e:
    raise HTTPException(status_code=500, detail=f"Failed to fetch EncodeFile.pkl from GitHub: {str(e)}")

def recognize_faces(image):
    img = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces_locations = face_recognition.face_locations(img)
    tmp_faces_enc = face_recognition.face_encodings(img, faces_locations)
    print(tmp_faces_enc)

    result = []
    for cur_enc_face in tmp_faces_enc:
        facesDistances = face_recognition.face_distance(trained_people, cur_enc_face)

        best_match = np.argmin(facesDistances)
        if facesDistances[best_match] < 0.6:
            result.append(names[best_match])

    return result

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    
    recognized_names = recognize_faces(image)

    # Save the result to a text file
    result_text = "\n".join(recognized_names)
    with open("result.txt", "w") as result_file:
        result_file.write(result_text)

    return FileResponse("result.txt")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
