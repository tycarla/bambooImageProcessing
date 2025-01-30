import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import torch
from pathlib import Path

# Load the YOLOv5 model with the specified path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\carla\Desktop\bambooImageProcessing\yolov5\runs\train\exp5\weights\best.pt')  # Specify full path to best.pt

# Global variables to store selected image path and processed image
selected_image_path = None
processed_image = None


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
    global processed_image
    if selected_image_path:
        # Perform detection using the model
        results = model(selected_image_path)
        results.save(save_dir='detected')  # Save the result in the "detected" folder
        processed_image_path = Path('detected') / Path(selected_image_path).name

        # Display the result
        img = Image.open(processed_image_path)
        img = img.resize((400, 400))  # Resize for display
        img = ImageTk.PhotoImage(img)
        display_image_label.config(image=img)
        display_image_label.image = img

        # Update buttons for the next step
        use_photo_button.pack_forget()
        choose_another_button.config(text="Choose Another Photo", command=choose_image)
        choose_another_button.pack(side=tk.BOTTOM, pady=10)


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

# Start the Tkinter event loop
root.mainloop()
