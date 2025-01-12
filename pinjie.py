from PIL import Image
import numpy as np
import os

# 定义图像的宽度和高度
image_width = 256
image_height = 256

# 定义大图的行列数
rows = 70
columns = 45

model_name="manet"

# 创建一个空白的大图
final_image = Image.new('RGB', (columns * image_width, rows * image_height))

# 读取图像并拼接到大图上
for i in range(rows):
    for j in range(columns):
        # 计算图像的索引
        index = i * columns + j
        if index < 3150:
            # 读取图像
            #image_path = f"dataset/image/{index}.png"
            #image_path = f"dataset/label/{index}.png"
            image_path = f"test_results_{model_name}/{index}.png"
            if os.path.exists(image_path):
                image = Image.open(image_path)
                # 将图像拼接到大图上
                final_image.paste(image, (j * image_width, i * image_height))

# 保存最终的大图
#final_image.save("final_image.png")
#final_image.save("final_label.png")
final_image.save(f"final_label_{model_name}.png")

