import gradio as gr
import torch
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image
import json
import os
# -------------------------------
# Load class names
# -------------------------------
with open("classes.json", "r") as f:
    class_names = json.load(f)

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load EfficientNet-B0
# -------------------------------
model = efficientnet_b0(weights=None)

# Replace the classifier exactly as during training
model.classifier[1] = torch.nn.Linear(1280, 5)

# Load trained weights
model.load_state_dict(torch.load("best_pet_disease_model.pth", map_location=device))

model.to(device)
model.eval()
# -------------------------------
# Image preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# ==========================================================
# Prediction Function
# ==========================================================

def predict_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Apply preprocessing
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    # Move to CPU/GPU
    image = image.to(device)

    # Prediction
    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # Convert probabilities to numpy
    probabilities = probabilities.cpu().numpy()[0]

    # Create dictionary for Gradio
    results = {}

    for i, disease in enumerate(class_names):

        results[disease] = float(probabilities[i])

    return results

# ==========================================================
# Create folders for user feedback
# ==========================================================

CORRECT_FOLDER = "correct_predictions"
INCORRECT_FOLDER = "incorrect_predictions"

os.makedirs(CORRECT_FOLDER, exist_ok=True)
os.makedirs(INCORRECT_FOLDER, exist_ok=True)

# ==========================================================
# Save correctly predicted images
# ==========================================================

def save_correct(image):

    filename = f"{CORRECT_FOLDER}/correct_{len(os.listdir(CORRECT_FOLDER))+1}.png"

    image.save(filename)

    return "✅ Thank you! Image saved as a correct prediction."
# ==========================================================
# Save incorrect predictions
# ==========================================================

def save_incorrect(image):

    filename = f"{INCORRECT_FOLDER}/incorrect_{len(os.listdir(INCORRECT_FOLDER))+1}.png"

    image.save(filename)

    return "❌ Image saved for future retraining."


# ==========================================================
# Build Gradio Interface
# ==========================================================

with gr.Blocks(title="Pet Disease Detection AI") as demo:

    gr.Markdown(
        """
        # 🐶 Pet Disease Detection AI

        Upload a picture of a dog or cat skin disease.

        The AI will predict the disease and display the confidence scores.
        """
    )

    with gr.Row():

        image_input = gr.Image(
            type="pil",
            label="Upload Pet Image"
        )

        prediction_output = gr.Label(
            num_top_classes=5,
            label="Predictions"
        )

    predict_button = gr.Button("🔍 Analyze Image", variant="primary")

    feedback_message = gr.Textbox(
        label="Feedback",
        interactive=False
    )

    with gr.Row():

        correct_button = gr.Button("👍 Prediction Correct")

        incorrect_button = gr.Button("👎 Prediction Incorrect")

# ==========================================================
# Connect Buttons
# ==========================================================

predict_button.click(
    fn=predict_image,
    inputs=image_input,
    outputs=prediction_output
)

correct_button.click(
    fn=save_correct,
    inputs=image_input,
    outputs=feedback_message
)

incorrect_button.click(
    fn=save_incorrect,
    inputs=image_input,
    outputs=feedback_message
)

# ==========================================================
# Launch App
# ==========================================================

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )