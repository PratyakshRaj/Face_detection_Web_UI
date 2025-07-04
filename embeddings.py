import cv2
import numpy as np
from insightface.app import FaceAnalysis

face_app = FaceAnalysis(name='buffalo_s')
face_app.prepare(ctx_id=0)

def extract_embedding(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    faces = face_app.get(img)
    return img, faces

def is_match(known_embedding, test_embedding, threshold=0.35):
    known_norm = known_embedding / np.linalg.norm(known_embedding)
    test_norm = test_embedding / np.linalg.norm(test_embedding)
    similarity = np.dot(known_norm, test_norm)
    return similarity > threshold, similarity
