# knn.py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

EPS = 1e-12


def l2_normalize(x: np.ndarray):
    # 벡터 정규화 불가 (NaN, 0벡터)
    n = np.linalg.norm(x)
    if not np.isfinite(n) or n < EPS:
        return None
    return x / n


class FullKNN:
    def __init__(self, embeddings_path: str):
        
        # 임베딩 데이터 로드
        df = pd.read_parquet(Path(embeddings_path))

        # 임베딩 벡터 행렬화
        X = np.vstack(df["vector"].apply(lambda v: np.asarray(v, np.float32)))
        norms = np.linalg.norm(X, axis=1)

        # NaN / zero-norm 벡터 제거
        ok = np.isfinite(X).all(axis=1) & (norms > EPS)
        df = df.loc[ok].reset_index(drop=True)
        X = X[ok] / norms[ok][:, None]

        # 임베딩 및 메타데이터 저장
        self.X = X
        self.gender = df["gender"].to_numpy()
        self.animal = df["animal_type"].to_numpy()

        # 성별별 인덱스 미리 구성
        self.idx_by_gender = {
            g: np.where(self.gender == g)[0]
            for g in np.unique(self.gender)
        }

    def predict(self, user_vec: np.ndarray, gender: str, K: int = 50):
        
        # 성별 데이터 x 중단
        if gender not in self.idx_by_gender:
            return []

        # 사용자 임베딩 정규화
        u = l2_normalize(np.asarray(user_vec, np.float32))
        if u is None:
            return []

        # 성별에 해당하는 임베딩만 사용
        idx = self.idx_by_gender[gender]
        Xg = self.X[idx]
        ag = self.animal[idx]

        # 코사인 유사도 계산
        sims = Xg @ u

        # top-K 이웃 선택
        K = min(K, len(sims))
        top_idx = np.argpartition(-sims, K - 1)[:K]

        # 동물상별 가중 투표 (유사도 합산)
        scores = defaultdict(float)
        for i in top_idx:
            s = sims[i]
            if s > 0:
                scores[ag[i]] += float(s)

        if not scores:
            return []

        # 가장 점수가 높은 동물상 반환
        best = max(scores.items(), key=lambda x: x[1])
        return [{"animal_type": best[0], "score": best[1]}]
