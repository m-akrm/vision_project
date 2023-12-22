from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import face_recognition
import cv2
import pickle
import numpy as np
import io
from PIL import Image
import tkinter as tk
from tkinter import filedialog

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load encoded faces and names
file = open('EncodeFile.pkl', 'rb')
encodeKnown = pickle.load(file)
file.close()
trained_people, names = encodeKnown


# Function to recognize faces
def recognize_faces(img):
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces_locations = face_recognition.face_locations(img)
    tmp_faces_enc = face_recognition.face_encodings(img, faces_locations)

    result_names = []
    for cur_enc_face in tmp_faces_enc:
        facesDistances = face_recognition.face_distance(trained_people, cur_enc_face)
        best_match = np.argmin(facesDistances)
        if facesDistances[best_match] < 0.6:
            result_names.append(names[best_match])

    return result_names


# Tkinter GUI
class GUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Face Recognition GUI")
        self.geometry("400x200")

        self.label = tk.Label(self, text="Choose an image to recognize faces:")
        self.label.pack(pady=10)

        self.button = tk.Button(self, text="Browse", command=self.browse_image)
        self.button.pack(pady=10)

        self.result_label = tk.Label(self, text="")
        self.result_label.pack(pady=10)

    def browse_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            image = cv2.imread(file_path)
            result_names = recognize_faces(image)
            self.result_label.config(text=f"Recognized Names: {', '.join(result_names)}")


# Define FastAPI routes
@app.get("/", response_class=HTMLResponse)
def read_root(request: dict):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result_names = recognize_faces(img)

    return {"filename": file.filename, "result_names": result_names}


if __name__ == "__main__":
    gui = GUI()
    gui.mainloop()
