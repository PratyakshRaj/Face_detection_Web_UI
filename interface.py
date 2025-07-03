import gradio as gr
import requests

def match_faces(reference_image, zip_file):
    if reference_image is None or zip_file is None:
        return "Please upload both the reference image and a ZIP of test images.", None

    files = [
        ("known_face", (reference_image.name, open(reference_image.name, "rb"), "image/jpeg")),
        ("test_zip", (zip_file.name, open(zip_file.name, "rb"), "application/zip")),
    ]

    response = requests.post("http://127.0.0.1:8000/compare/", files=files)

    if response.status_code == 200:
        zip_path = "matched_faces.zip"
        with open(zip_path, "wb") as f:
            f.write(response.content)
        return "✅ Match complete!", zip_path
    else:
        return f"❌ Error {response.status_code}: {response.text}", None

gr.Interface(
    fn=match_faces,
    inputs=[
        gr.File(label="Upload Reference Image", file_types=["image"]),
        gr.File(label="Upload ZIP of Test Images", file_types=[".zip"])
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.File(label="Download ZIP of Matching Faces")
    ],
    title="Face Matching Tool (ZIP Version)",
    description="Upload a reference image and a ZIP of test images. It will return a ZIP of all matching faces."
).launch()
