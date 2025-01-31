import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from yolov5.models.common import ClassificationModel  # Import YOLOv5 classification model

# Set the model path
MODEL_PATH = "D:/Programming/Thesis Data/Classification Test 1/yolov5/runs/train-cls/exp4/weights/best.pt"

# Load the trained classification model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClassificationModel(MODEL_PATH).to(device)
model.eval()  # Set to evaluation mode

# Define image preprocessing function
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match YOLOv5 classification input size
    transforms.ToTensor(),          # Convert to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Class labels (Modify based on your dataset)
CLASS_NAMES = ["Clumping", "Running"]

# Global variables
selected_image_path = None
prediction_label = None

# Function to choose an image
def choose_image():
    global selected_image_path, prediction_label
    selected_image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    
    if selected_image_path:
        # Display the chosen image
        img = Image.open(selected_image_path)
        img = img.resize((400, 400))
        img = ImageTk.PhotoImage(img)
        display_image_label.config(image=img)
        display_image_label.image = img

        # Remove previous prediction text
        if prediction_label:
            prediction_label.config(text="")

        # Show the "Classify" button
        classify_button.pack(pady=10)

# Function to classify the image
def classify_image():
    global prediction_label
    if selected_image_path:
        # Load and preprocess the image
        img = Image.open(selected_image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Perform inference
        with torch.no_grad():
            output = model(img)
            predicted_class = torch.argmax(output, dim=1).item()

        # Get predicted label
        predicted_label = CLASS_NAMES[predicted_class]

        # Display the classification result
        prediction_label = tk.Label(root, text=f"Prediction: {predicted_label}", font=("Arial", 14, "bold"), fg="blue")
        prediction_label.pack(pady=10)

# Initialize Tkinter window
root = tk.Tk()
root.title("Bamboo Growth Habit Classification")
root.geometry("600x700")

# Title Label
title_label = tk.Label(root, text="BAMBOO GROWTH HABIT CLASSIFICATION", font=("Arial", 18), pady=20)
title_label.pack()

# Image Display Label
display_image_label = tk.Label(root)
display_image_label.pack()

# Choose Photo Button
choose_photo_button = tk.Button(root, text="Choose Photo", command=choose_image, font=("Arial", 12), padx=10, pady=5)
choose_photo_button.pack(pady=20)

# Classify Button (Initially hidden)
classify_button = tk.Button(root, text="Classify Image", command=classify_image, font=("Arial", 12), padx=10, pady=5)
classify_button.pack_forget()

# Start the Tkinter event loop
root.mainloop()
