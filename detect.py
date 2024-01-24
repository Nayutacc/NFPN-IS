import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from model.ctNet.ctNet import Nest_U_Net
from torchvision.transforms import ToTensor

transf = ToTensor()

# 读取模型
model = Nest_U_Net()
model.load_state_dict(torch.load(r"save_model\best.pth", map_location=torch.device('cpu')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def preprocess(img_path):
    _img = Image.open(img_path)
    gray = _img.convert('L')  # convert to grayscale
    _img = np.array(gray)
    img_tensor = transf(_img)
    return img_tensor


with torch.no_grad():
    img = preprocess("./test/002103.png")
    img = img.reshape(1, 1, 256, 256)
    img = img.to(device)
    pred = model(img)
pred = pred.detach().cpu().numpy()  # 转为numpy数组
pred = pred.reshape(256, 256)


# 阈值分割
for row in range(0, 256):
    for col in range(0, 256):
        if pred[row, col] < 0.7:
            pred[row, col] = 0
        else:
            pred[row, col] = 1

plt.imshow(pred)
plt.show()
