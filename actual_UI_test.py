import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from pathlib import Path

# Load the YOLOv5 model with the specified path (use pretrained=False for custom model weights)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)  # Load YOLOv5s architecture
model.load_state_dict(torch.load(r'C:\Users\carla\Desktop\bambooImageProcessing\yolov5\runs\train\exp\weights\best.pt')['model'].float().state_dict())  # Load custom weights

# Set the model to evaluation mode
model.eval()

# Check if CUDA is available and move model to GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Global variables to store selected image path and processed image
selected_image_path = None
processed_image = None
result_image_path = None

# Function to choose an image
def choose_image():
    global selected_image_path, processed_image
    selected_image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if selected_image_path:
        # Display the chosen image
        img = Image.open(selected_image_path)
        img = img.resize((400, 400))  # Resize for display
        img = ImageTk.PhotoImage(img)
        display_image_label.config(image=img)
        display_image_label.image = img
        processed_image = None  # Reset processed image
        use_photo_button.pack(side=tk.LEFT, padx=10)
        choose_another_button.pack(side=tk.RIGHT, padx=10)

# Function to process the image with YOLOv5
def use_this_photo():
    global processed_image, result_image_path
    if selected_image_path:
        # Perform detection using the model
        img = Image.open(selected_image_path)  # Open image for detection
        img_tensor = torch.from_numpy(np.array(img))  # Convert to numpy array and then tensor
        img_tensor = img_tensor.permute(2, 0, 1)  # Rearrange dimensions to match model input
        img_tensor = img_tensor.unsqueeze(0).float().to(device)  # Add batch dimension and move to GPU

        # Run inference
        results = model(img_tensor)  # Perform inference
        
        # Save the result in the "detected" folder
        results.save(save_dir='detected')  
        result_image_path = Path('detected') / Path(selected_image_path).name

        # Display the result
        img = Image.open(result_image_path)
        img = img.resize((400, 400))  # Resize for display
        img = ImageTk.PhotoImage(img)
        display_image_label.config(image=img)
        display_image_label.image = img

        # Show the detected image and update buttons
        use_photo_button.pack_forget()
        choose_another_button.config(text="Choose Another Photo", command=choose_image)
        choose_another_button.pack(side=tk.BOTTOM, pady=10)

# Function to reset the application to choose a new image
def reset_app():
    choose_image()
    choose_another_button.pack_forget()  # Hide "Choose Another" button initially

# Initialize the main Tkinter window
root = tk.Tk()
root.title("Bamboo Growth Habit Classification")
root.geometry("600x600")

# Title Label
title_label = tk.Label(root, text="BAMBOO GROWTH HABIT CLASSIFICATION", font=("Arial", 18), pady=20)
title_label.pack()

# Image Display Label
display_image_label = tk.Label(root)
display_image_label.pack()

# Choose Photo Button
choose_photo_button = tk.Button(root, text="Choose Photo", command=choose_image, font=("Arial", 12), padx=10, pady=5)
choose_photo_button.pack(pady=20)

# Buttons for "Use this photo" and "Choose another photo"
use_photo_button = tk.Button(root, text="Use This Photo", command=use_this_photo, font=("Arial", 12), padx=10, pady=5)
choose_another_button = tk.Button(root, text="Choose Another Photo", command=choose_image, font=("Arial", 12), padx=10, pady=5)

# Initially hide the "Choose Another" button
choose_another_button.pack_forget()

# Start the Tkinter event loop
root.mainloop()
