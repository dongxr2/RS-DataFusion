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
import torch.nn.functional as F

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

# 调整颜色列表，使其包含一致的RGB值
#label_colors = np.array(label_colors, dtype=np.uint8)

# 定义测试数据集类
class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'image')
        self.image_files = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])

        image = Image.open(img_name).convert('RGB')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image)

        return image

# 定义模型
class DLModel(nn.Module):
    def __init__(self):
        super(DLModel, self).__init__()
        #self.model = smp.Unet('resnet18', in_channels=3, classes=10)  # 修改为10分类
        #self.model = smp.UnetPlusPlus('resnet18', in_channels=3, classes=10)  # 修改为10分类
        #self.model = smp.DeepLabV3('resnet18', in_channels=3, classes=10) 
        #self.model = smp.DeepLabV3Plus('resnet18', in_channels=3, classes=10)  # 修改为10分类
        #self.model = smp.DeepLabV3PWT('resnet18', in_channels=3, classes=10) 
        #self.model=smp.PSPNet('resnet18', in_channels=3, classes=10)
        self.model=smp.MAnet('resnet18',in_channels=3,classes=10)
        #self.model=smp.FPN('resnet18',in_channels=3,classes=10)
        #self.model=smp.PAN('resnet18',in_channels=3,classes=10)
        #self.model=smp.Linknet('resnet18',in_channels=3,classes=10)
        #self.model = smp.EffiTUnet('resnet18', in_channels=3, classes=10)
        #self.model=smp.UnetPlusPlusWT('resnet18', in_channels=3, classes=10)
        #self.model=smp.PSPNetWT('resnet18',in_channels=3,classes=10)

    def forward(self, x):
        return self.model(x)

model_name='manet'
# 定义模型和加载训练好的权重
model = DLModel()
model.load_state_dict(torch.load(f'rs_{model_name}.pth'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 预测函数
def predict(model, test_loader, dataset):
    outputs_list = []
    filenames_list = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            filenames_list.extend(dataset.image_files)

            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            outputs_list.append(outputs.cpu())

    return outputs_list, filenames_list

# 设置测试参数
root_dir = 'dataset'
batch_size = 1

# 准备测试数据
test_dataset = TestDataset(root_dir)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 进行预测
outputs_list, filenames_list = predict(model, test_loader, test_dataset)



# 保存预测结果
output_dir = f'test_results_{model_name}'
os.makedirs(output_dir, exist_ok=True)

for idx, (outputs, filename) in enumerate(zip(outputs_list, filenames_list)):
    pred = torch.argmax(outputs, dim=1).detach().numpy()[0]
    pred_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

    for i, color in enumerate(label_colors):
        pred_color[pred == i] = color

    pred_image = Image.fromarray(pred_color)
    # 使用输入图像的文件名进行命名
    pred_image.save(os.path.join(output_dir, f'{filename}'))

