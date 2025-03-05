import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from model.ctNet.ctNet import Nest_U_Net
from torchvision.transforms import ToTensor
transf = ToTensor()

# 读取模型
model = Nest_U_Net()
checkpoint = torch.load(r"best_model_epoch_137.pth", map_location=lambda storage, loc: storage, weights_only=True)
model.load_state_dict(checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def preprocess(img_path):
    _img = Image.open(img_path)
    gray = _img.convert('L')  # convert to grayscale
    _img = np.array(gray)
    img_tensor = transf(_img)
    return img_tensor


def detect(image_path):
    with torch.no_grad():
        img = preprocess(image_path)
        img = img.reshape(1, 1, 256, 256)
        img = img.to(device)
        pred = model(img)
    pred = pred.detach().cpu().numpy()  # 转为numpy数组
    pred = pred.reshape(256, 256)
    mask_img = Image.fromarray((pred * 255).astype(np.uint8)).convert("L")
    
    mask_img = Image.fromarray((pred * 255).astype(np.uint8))
    mask_img.save('result.png')
    plt.imshow(mask_img)
    plt.show()


if __name__ == '__main__':
    image_path = r""
    detect(image_path)
