from model.ctNet.DataLoader import MyDataset
from model.ctNet.ctNet import Nest_U_Net
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


# Function to compute AUC for a batch of images and masks
def compute_batch_auc(_predictions, _masks):
    # Assuming the model's output is in the form (batch_size, height, width)
    _predictions = _predictions.flatten()
    _masks = _masks.flatten()
    return roc_auc_score(_masks, _predictions)


train_txt_path = "D:/work/python/dataset/NUDT/train.txt"  # path to your train.txt file
test_txt_path = "D:/work/python/dataset/NUDT/test.txt"
image_base_path = "D:/work/python/dataset/NUDT/images"  # Replace with the actual image directory path
mask_base_path = "D:/work/python/dataset/NUDT/masks"  # Replace with the actual mask directory path

mydataset_train = MyDataset(train_txt_path, image_base_path, mask_base_path)  # 训练集数据
mydataset_test = MyDataset(test_txt_path, image_base_path, mask_base_path)  # 测试集数据
data_train = DataLoader(mydataset_train, batch_size=32, shuffle=True)
data_test = DataLoader(mydataset_test, batch_size=32, shuffle=False)

mymodule = Nest_U_Net()  # 实例化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mymodule = mymodule.to(device)
loss_fn = nn.MSELoss()  # 实例化损失函数
optimizer = optim.Adam(mymodule.parameters(), lr=0.001)  # 实例化优化器

epoch = 30  # 设置迭代次数

if __name__ == '__main__':
    for i in range(epoch):
        print("--------第{}次迭代--------".format(i))
        for index, (img, mask) in enumerate(data_train):
            print("---第{}组数据---".format(index))
            img = img.to(device)
            mask = mask.to(device)
            out = mymodule(img)
            loss = loss_fn(out, mask)
            optimizer.zero_grad()
            loss.backward()
            print("loss:", loss)
            with open("loss/loss.log", "a+", encoding="utf-8") as f:
                f.write(str(loss.item()) + "\n")
                f.close()
            optimizer.step()
        print("--------保存第{}次训练的模型为:model_epoch_{}.pth--------".format(i, i))
        torch.save(mymodule.state_dict(), "./save_model/23_10_8/model_epoch_{}.pth".format(i))

        mymodule.eval()  # Set the model to evaluation mode
        all_aucs = []
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(data_test):
                # Assuming the model's input expects tensors of size (batch_size, channels, height, width)
                images = images.to(device)
                predictions = mymodule(images)
                batch_auc = compute_batch_auc(predictions.cpu().numpy(), masks.cpu().numpy())
                all_aucs.append(batch_auc)
                if (batch_idx + 1) % 10 == 0:
                    with open("./logs/" + "epoch_{}".format(i) + "_AUC" + ".txt", "a+", encoding="utf-8") as f:
                        f.write(f"Batch {batch_idx + 1}/{len(data_test)} - AUC: {batch_auc:.7f}" + "\n")
                    print(f"Batch {batch_idx + 1}/{len(data_test)} - AUC: {batch_auc:.7f}")

        # Calculate the average AUC
        mean_auc = np.mean(all_aucs)
        with open("./logs/all_mean_auc.txt", "a+", encoding="utf-8") as f:
            f.write("Average AUC of " + "epoch_{}".format(i) + ":{:.7f}".format(mean_auc) + "\n")

        mymodule.train()  # Set the model back to training mode
