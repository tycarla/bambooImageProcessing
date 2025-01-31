import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import torch.nn.functional as F

# Load the trained model
model_path = r"D:\Programming\Thesis Data\Classification Test 1\yolov5\runs\train-cls\exp4\weights\best.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define class names (adjust based on dataset)
class_names = ["Running Bamboo", "Clumping Bamboo"]

# Initialize Tkinter
root = tk.Tk()
root.title("Bamboo Growth Habit Classification")
root.geometry("500x600")

# Variables
selected_image_path = None
img_label = None

# Function to open file dialog and select image
def choose_image():
    global selected_image_path, img_label
    
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    
    if file_path:
        selected_image_path = file_path
        
        # Display the selected image
        img = Image.open(selected_image_path)
        img.thumbnail((300, 300))  # Resize image for display
        img = ImageTk.PhotoImage(img)
        
        if img_label:
            img_label.config(image=img)
            img_label.image = img
        else:
            img_label = tk.Label(root, image=img)
            img_label.image = img
            img_label.pack()

        # Show buttons to confirm or change image
        confirm_btn.pack()
        change_btn.pack()

# Function to run classification
def classify_image():
    global selected_image_path
    
    if not selected_image_path:
        return
    
    # Load and preprocess the image
    image = Image.open(selected_image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0).to(device)

    # Perform classification
    results = model(image)
    probs = F.softmax(results, dim=1)
    predicted_class_idx = torch.argmax(probs, dim=1).item()
    predicted_label = class_names[predicted_class_idx]

    # Convert confidence to percentage
    confidence = probs[0][predicted_class_idx].item() * 100

    # Display classification result
    result_label.config(text=f"Predicted: {predicted_label}\nConfidence: {confidence:.2f}%")
    result_label.pack()

# UI Elements
title_label = tk.Label(root, text="Bamboo Growth Habit Classification", font=("Arial", 16, "bold"))
title_label.pack(pady=20)

choose_btn = tk.Button(root, text="Choose Image", command=choose_image, font=("Arial", 12))
choose_btn.pack()

confirm_btn = tk.Button(root, text="Use This Image", command=classify_image, font=("Arial", 12))
change_btn = tk.Button(root, text="Choose Another Image", command=choose_image, font=("Arial", 12))

result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))

# Start the application
root.mainloop()
