# preprocess.py
import numpy as np
import cv2


def biggest_face(faces):
    # 여러 얼굴 중 가장 큰 얼굴 선택
    return max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )


def safe_norm_vec(vec: np.ndarray):
    # 임베딩 벡터 정규화 (NaN / 0벡터 방어)
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-12:
        return None
    return v / n


def decode_upload_image(img_bytes: bytes):
    # 업로드된 이미지 byte → OpenCV 이미지 변환
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    return img
