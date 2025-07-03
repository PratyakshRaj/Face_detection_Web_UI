from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os, shutil, zipfile, uuid, tempfile
from embeddings import extract_embedding, is_match
from PIL import Image

app = FastAPI()

def is_image(file_path):
    try:
        Image.open(file_path).verify()
        return True
    except:
        return False

@app.post("/compare/")
async def compare_faces(
    known_face: UploadFile = File(...),
    test_zip: UploadFile = File(...)
):
    # Step 1: Read and process known face
    known_bytes = await known_face.read()
    known_img, known_faces = extract_embedding(known_bytes)
    if not known_faces:
        return {"error": "No face found in reference image."}
    known_embedding = known_faces[0].embedding

    # Step 2: Extract ZIP to temp dir
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "test_images.zip")
    with open(zip_path, "wb") as f:
        f.write(await test_zip.read())

    extract_dir = os.path.join(temp_dir, "unzipped")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Step 3: Compare each extracted image
    match_dir = os.path.join(temp_dir, f"matches_{uuid.uuid4().hex}")
    os.makedirs(match_dir, exist_ok=True)

    for root, _, files in os.walk(extract_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            if not is_image(file_path):
                continue

            try:
                with open(file_path, "rb") as f:
                    img_bytes = f.read()

                test_img, test_faces = extract_embedding(img_bytes)
                if not test_faces:
                    continue

                for face in test_faces:
                    match, similarity = is_match(known_embedding, face.embedding)
                    if match:
                        shutil.copy(file_path, os.path.join(match_dir, file_name))
                        break
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

    # Step 4: Zip matched results
    zip_result = os.path.join(temp_dir, "matched_faces.zip")
    with zipfile.ZipFile(zip_result, 'w') as zipf:
        for file in os.listdir(match_dir):
            zipf.write(os.path.join(match_dir, file), arcname=file)

    # Step 5: Return zipped file
    return FileResponse(zip_result, filename="matched_faces.zip", media_type="application/zip")
