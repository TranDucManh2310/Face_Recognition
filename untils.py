import face_recognition
import os

def load_known_face(file_path):
    # Nạp ảnh từ đường dẫn file
    known_image = face_recognition.load_image_file(file_path)
    
    # Trích xuất mã hóa khuôn mặt từ ảnh
    face_encodings = face_recognition.face_encodings(known_image)
    
    # Nếu không tìm thấy khuôn mặt nào trong ảnh, ném lỗi
    if not face_encodings:
        raise ValueError("No face found in the image!")
    
    # Tạo danh sách nhãn cho từng khuôn mặt (ví dụ: "Known Face 1", "Known Face 2", ...)
    # Bạn cũng có thể tùy chỉnh nhãn dựa trên tên file hoặc các tiêu chí khác
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    known_face_labels = [f"{name_without_ext} {i+1}" for i in range(len(face_encodings))]
    
    # Trả về danh sách mã hóa khuôn mặt và danh sách nhãn
    return face_encodings, known_face_labels
