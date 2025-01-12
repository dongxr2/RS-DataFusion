import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# 定义颜色列表
label_colors = [
    (254, 246, 201),   # 耕地
    (16, 118, 75),     # 森林
    (172, 212, 90),    # 草地
    (57, 179, 115),    # 灌木地
    (124, 209, 245),   # 湿地
    (0, 87, 155),      # 水体
    (96, 102, 48),     # 苔原
    (147, 45, 16),     # 人造地表
    (206, 203, 206),   # 裸土
    (214, 242, 255)    # 冰川和永久积雪
]
#label_colors = [tuple(reversed(color)) for color in label_colors]

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'image')
        self.mask_dir = os.path.join(root_dir, 'label')
        self.image_files = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        mask_name = os.path.join(self.mask_dir, self.image_files[idx])

        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('RGB')  # 提取RGB标签图

        # 处理RGB标签图为索引形式
        mask = np.array(mask)
        mask_index = np.zeros(mask.shape[:2], dtype=np.uint8)
        for i, color in enumerate(label_colors):
            mask_index[(mask == color).all(axis=2)] = i
        
        '''unique_values, counts = np.unique(mask_index, return_counts=True)
        count_sum=0
        # 输出每个值及其对应个数
        for value, count in zip(unique_values, counts):
            print(f"Value {value}: {count} times")
            count_sum+=count
        print(f"Total: {count_sum}")
        input()'''

        mask_one_hot = np.eye(len(label_colors))[mask_index]
        mask_tensor = torch.tensor(mask_one_hot.transpose(2, 0, 1), dtype=torch.float32)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image)

        return image, mask_tensor

# 定义模型
class DLModel(nn.Module):
    def __init__(self):
        super(DLModel, self).__init__()
        #self.model = smp.Unet('resnet18', in_channels=3, classes=10)  
        #self.model = smp.UnetPlusPlus('resnet18', in_channels=3, classes=10) 
        #self.model = smp.DeepLabV3('resnet18', in_channels=3, classes=10) 
        #self.model = smp.DeepLabV3Plus('resnet18', in_channels=3, classes=10) 
        #self.model = smp.MAnet('resnet18', in_channels=3, classes=10) 
        #self.model = smp.DeepLabV3PWT('resnet18', in_channels=3, classes=10) 
        #self.model=smp.PSPNet('resnet18', in_channels=3, classes=10)
        #self.model=smp.FPN('resnet18',in_channels=3,classes=10)
        #self.model=smp.PAN('resnet18',in_channels=3,classes=10)
        #self.model=smp.Linknet('resnet18',in_channels=3,classes=10)
        #-----------self.model = smp.UnetTB('resnet18', in_channels=3, classes=10)  
        self.model = smp.EffiTUnet('resnet18', in_channels=3, classes=10)
        #self.model=smp.UnetPlusPlusWT('resnet18', in_channels=3, classes=10)
        #self.model=smp.PSPNetWT('resnet18',in_channels=3,classes=10)

    def forward(self, x):
        return self.model(x)

# 训练函数
import torch
import matplotlib.pyplot as plt

def train(model, train_loader, criterion, optimizer, num_epochs=10, save_loss_path=None, save_model_path=None, save_best_only=True, save_interval=1):
    model.train()
    loss_values = []
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, masks in train_loader:
            inputs = inputs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        loss_values.append(epoch_loss)
        
        # 保存loss图像
        if save_loss_path:
            plt.plot(loss_values, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.savefig(save_loss_path)
            plt.close()
        
        # 动态保存模型
        if save_model_path:
            # 保存每隔`save_interval`个epoch的模型
            if not save_best_only and (epoch + 1) % save_interval == 0:
                torch.save(model.state_dict(), f"{save_model_path}_epoch_{epoch+1}.pth")
            
            # 只保存loss最低的模型
            if save_best_only and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), f"{save_model_path}_best.pth")


# 设置训练参数
root_dir = 'dataset'
batch_size = 128 #pspnet/wt 64，others 128
learning_rate = 0.001
num_epochs = 100
model_keyword="manet"
save_loss_path = f'training_loss_plot_rs_{model_keyword}.png'
save_model_path = f'rs_{model_keyword}.pth'

# 准备数据
dataset = CustomDataset(root_dir)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DLModel().to(device)
criterion = nn.BCEWithLogitsLoss()  # 多类别损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
train(model, train_loader, criterion, optimizer, num_epochs=num_epochs, save_loss_path=save_loss_path, save_model_path=save_model_path)
