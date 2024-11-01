import torch
import timm

# Load ZoeDepth model
torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo

repo = "isl-org/ZoeDepth"
# Zoe_N
model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

#### SAMPLE PREDICTION USING ZOEDEPTH ####
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

# Load image
from PIL import Image
image = Image.open("photos/image.jpg")
print("Image Size:", image.size)

# As Numpy
depth_np = zoe.infer_pil(image)

# As 16-bit PIL image
depth_pil = zoe.infer_pil(image, output_type="pil")

# As Torch Tensor
depth_torch = zoe.infer_pil(image, output_type="tensor")


# print("ZoeDepth Numpy:", depth_np.shape)
# print("ZoeDepth PIL:", depth_pil.size)
# print("ZoeDepth Torch:", depth_torch.shape)

# Save raw
from ZoeDepth.zoedepth.utils.misc import save_raw_16bit
fpath = "depth_output.png"
save_raw_16bit(depth_np, fpath)

# Colorize output
from ZoeDepth.zoedepth.utils.misc import colorize

colored = colorize(depth_np)

# save colored output
fpath_colored = "depth_output_colored.png"
Image.fromarray(colored).save(fpath_colored)

# Object detection using YOLOv11
from ultralytics import YOLO
yolo = YOLO("yolo11x.pt")

# Train the model
# train_results = yolo.train(
#     data="coco8.yaml",  # path to dataset YAML
#     epochs=100,  # number of training epochs
#     imgsz=640,  # training image size
#     device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
# )

# Evaluate model performance on the validation set
#metrics = yolo.val()


# Perform object detection on an image
results = yolo("photos/image.jpg")

import cv2
all_bottom_centers = []
top_left_corners = []
bottom_right_corners = []
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs

    # Display and save result image
    result.show()  # Display on screen
    result.save(filename="result.jpg")  # Save to disk

    # Calculate bottom center coordinates for each bounding box in the current result
    bottom_centers = torch.stack([
        boxes.xywh[:, 0],  # x_center remains the same
        boxes.xywh[:, 1] + boxes.xywh[:, 3] / 2  # y_center + height / 2
    ], dim=1)

    # Top-left corner of the bounding box
    top_left = torch.stack([
        boxes.xywh[:, 0] - boxes.xywh[:, 2] / 2,  # x_center - width / 2
        boxes.xywh[:, 1] - boxes.xywh[:, 3] / 2  # y_center - height / 2
    ], dim=1)

    bottom_right = torch.stack([
        boxes.xywh[:, 0] + boxes.xywh[:, 2] / 2,  # x_center + width / 2
        boxes.xywh[:, 1] + boxes.xywh[:, 3] / 2  # y_center + height / 2
    ], dim=1)

    # Convert bottom_centers to integers for OpenCV and add to list
    all_bottom_centers.extend(bottom_centers.int().tolist())
    top_left_corners.extend(top_left.int().tolist())
    bottom_right_corners.extend(bottom_right.int().tolist())


print("Bottom Centers:", all_bottom_centers)
print(top_left, bottom_right)
print(depth_np.max())


# Depth map values for the bounding box






# Make a copy of the original image
img_with_dots = results[0].orig_img.copy()  # Assuming all results share the same original image

# Draw a dot at the bottom center of each bounding box collected from all results
radius = 2
color = (0, 255, 0)  # Green color for the dot
thickness = -1       # Filled circle

for bottom_center in all_bottom_centers:
    x, y = bottom_center[0], bottom_center[1]
    img_with_dots = cv2.circle(img_with_dots, (x, y), radius, color, thickness)

# 

# Show the image with the dots
cv2.imshow('Image with Dots', img_with_dots)
cv2.waitKey(0)
cv2.destroyAllWindows()
