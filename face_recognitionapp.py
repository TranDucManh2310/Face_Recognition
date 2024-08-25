import cv2
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from untils import load_known_face  # Đảm bảo rằng tên tệp là 'utils.py'

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")
        
        self.video_capture = cv2.VideoCapture(0)
        
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()
        
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(fill=tk.X, pady=10)
        
        self.load_btn = tk.Button(self.btn_frame, text="Load Known Face", command=self.load_known_face)
        self.load_btn.pack(side=tk.LEFT, padx=10)
        
        self.quit_btn = tk.Button(self.btn_frame, text="Quit", command=root.quit)
        self.quit_btn.pack(side=tk.RIGHT, padx=10)
        
        self.known_face_encodings = []
        self.known_face_labels = []
        
        self.update_frame()
    
    def load_known_face(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            encodings, labels = load_known_face(file_path)
            self.known_face_encodings.extend(encodings)  # Thêm tất cả các mã hóa vào danh sách
            self.known_face_labels.extend(labels)  # Thêm tất cả các nhãn vào danh sách
    
    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                label = "Unknown Face"
                if self.known_face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    
                    if matches:  # Kiểm tra nếu matches không rỗng
                        best_match_index = np.argmin(face_distances)
                        
                        # Kiểm tra nếu best_match_index hợp lệ và không vượt quá phạm vi
                        if best_match_index < len(matches) and matches[best_match_index]:
                            label = self.known_face_labels[best_match_index]
                
                color = (0, 255, 0) if label != "Unknown Face" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk
        
        self.root.after(10, self.update_frame)
    
    def __del__(self):
        self.video_capture.release()
