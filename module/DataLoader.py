from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import ToTensor
import os

transf = ToTensor()


def preprocess(img_path):
    img = Image.open(img_path)
    gray = img.convert('L')  # convert to grayscale
    img = np.array(gray)
    img_tensor = transf(img)
    return img_tensor


class MyDataset(Dataset):
    """
    读取img，和mask转换为tensor返回
    """

    def __init__(self, index_path, image_base_path, mask_base_path):
        self.index_path = index_path
        self.image_base_path = image_base_path
        self.mask_base_path = mask_base_path
        with open(index_path, encoding="utf-8") as f:
            self.img_idx = f.read().splitlines()

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, index):
        img_filename = self.img_idx[index]
        img_path = os.path.join(self.image_base_path, img_filename)
        mask_path = os.path.join(self.mask_base_path, img_filename)

        img = preprocess(img_path)
        mask = preprocess(mask_path)
        return img, mask
