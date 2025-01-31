import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# Define class names (adjust based on your dataset's labels)
class_names = ["Running Bamboo", "Clumping Bamboo"]

# Load the trained model
model_path = r"D:\Programming\Thesis Data\Classification Test 1\yolov5\runs\train-cls\exp4\weights\best.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Set image path (Replace with actual image file path)
image_path = r"D:\Programming\Thesis Data\Classification Test 1\Testing Images\IMG_2867.jpg"

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")  # Open image in RGB mode

# Define transformation to match YOLOv5 classification input format
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match YOLOv5 classification input size
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

image = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)

# Move image to same device as model
image = image.to(device)

# Perform classification
results = model(image)

# Apply softmax to get probabilities
probs = F.softmax(results, dim=1)

# Get the predicted class index
predicted_class_idx = torch.argmax(probs, dim=1).item()

# Get the class label
predicted_label = class_names[predicted_class_idx]

# Print results
print(f"Predicted Class: {predicted_label}")
print(f"Confidence Scores: {probs.tolist()}")
