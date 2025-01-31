import torch

# Load the YOLOv5 model with custom weights (best.pt) for inference
model = torch.hub.load('ultralytics/yolov5:v5.0', 'yolov5s')  # Load a YOLOv5 model (small version)

# Load custom weights
model.load_state_dict(torch.load(r'D:\Programming\Thesis Data\Classification Test 1\yolov5\runs\train-cls\exp4\weights\best.pt')['model'].float().state_dict())

model.eval()  # Set the model to evaluation mode

model.eval()  # Set the model to evaluation mode

# To perform inference, pass an image through the model (use an image in the correct format)
img = 'path_to_image.jpg'  # Replace with the path to your image
results = model(img)  # Run inference
results.show()  # Display the results
