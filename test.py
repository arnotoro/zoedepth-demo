import torch
import timm

# Load ZoeDepth model
torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=False)  # Triggers fresh download of MiDaS repo

repo = "isl-org/ZoeDepth"
# Zoe_N
model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

#### SAMPLE PREDICTION USING ZOEDEPTH ####
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

# Load image
from PIL import Image
image = Image.open("image4.jpg")
print("Image Size:", image.size)

# As Numpy
depth_np = zoe.infer_pil(image)
