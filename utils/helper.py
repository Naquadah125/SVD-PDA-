import base64
import numpy as np
import cv2

def decode_image(contents):
    """
    Chuyển chuỗi Base64 từ Web thành ảnh OpenCV (Grayscale).
    """
    if contents is None:
        return None
    try:
        # Tách phần header "data:image/..."
        encoded_data = contents.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        # Đọc ảnh xám để xử lý ma trận 2D
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None

def encode_image(img_array):
    """
    Chuyển ảnh OpenCV (numpy array) ngược lại thành Base64 để hiển thị lên Web.
    """
    _, buffer = cv2.imencode('.jpg', img_array)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{jpg_as_text}"