# main.py
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from insightface.app import FaceAnalysis

from knn import FullKNN
from preprocess import biggest_face, safe_norm_vec, decode_upload_image


# ===== 설정 =====
EMB_PATH = "./artifacts/embeddings_fixed.parquet"
DET_SIZE = (640, 640)


# ===== 서버 시작 시 로딩 =====
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=DET_SIZE)

knn = FullKNN(EMB_PATH)

app = FastAPI()


@app.post("/analyze")
async def analyze(
    gender: str = Form(...),
    file: UploadFile = File(...),
):
    # 남녀 여부
    if gender not in ("남자", "여자"):
        raise HTTPException(400, "gender must be 남자 or 여자")

    # 사진 여부
    img_bytes = await file.read()
    img = decode_upload_image(img_bytes)
    if img is None:
        raise HTTPException(400, "invalid image")

    # 얼굴 여부
    faces = face_app.get(img)
    if not faces:
        raise HTTPException(422, "no face detected")

    f = biggest_face(faces)

    # 얼굴 선명도
    det_score = float(getattr(f, "det_score", 0.0))
    if det_score < 0.5:
        raise HTTPException(422, "face too unclear")

    # 임베딩 불가
    vec = safe_norm_vec(np.asarray(f.embedding, dtype=np.float32))
    if vec is None:
        raise HTTPException(422, "invalid embedding")

    # === knn ===
    result = knn.predict(
        user_vec=vec,
        gender=gender,
        K=50
    )

    if not result:
        raise HTTPException(422, "no prediction")

    return {
        "animal_type": result[0]["animal_type"],
        "det_score": det_score,
    }
