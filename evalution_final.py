import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import os

np.set_printoptions(threshold=np.inf)
# 定义标签颜色和类别
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

num_classes = len(label_colors)

model_names = ['top3','top5']#["unet", "unetpp",'manet', "EffiTUnet", "deeplabv3", "deeplabv3p", "fpn", "linknet", "pan", "pspnet"]
average_modes = ["macro", "micro", "weighted"]

# 创建一个空的 DataFrame 列表，用于保存所有的结果
all_results = []

for model_name in model_names:
    for average_mode in average_modes:
        try:
            # 读取标签图和预测结果图
            label_path = "final_evalution/label/"
            result_path = f"final_evalution/{model_name}/"

            def read_images(image_path):
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image

            def load_data(path, image_names):
                images = []
                for name in image_names:
                    image = read_images(path + name)
                    images.append(image)
                return np.array(images)

            label_images = load_data(label_path, os.listdir(label_path))
            result_images = load_data(result_path, os.listdir(result_path))

            # 将RGB图像转换为类别索引图像
            def rgb_to_index(image, label_colors):
                index_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                for i, color in enumerate(label_colors):
                    index_image[np.all(image == color, axis=-1)] = i
                return index_image

            label_index_images = [rgb_to_index(image, label_colors) for image in label_images]
            result_index_images = [rgb_to_index(image, label_colors) for image in result_images]

            # 计算性能指标
            accuracy = [accuracy_score(label.flatten(), result.flatten()) for label, result in zip(label_index_images, result_index_images)]
            precision = [precision_score(label.flatten(), result.flatten(), average=average_mode, zero_division=0) for label, result in zip(label_index_images, result_index_images)]
            recall = [recall_score(label.flatten(), result.flatten(), average=average_mode, zero_division=0) for label, result in zip(label_index_images, result_index_images)]
            f1 = [f1_score(label.flatten(), result.flatten(), average=average_mode, zero_division=0) for label, result in zip(label_index_images, result_index_images)]
            iou = [jaccard_score(label.flatten(), result.flatten(), average=average_mode) for label, result in zip(label_index_images, result_index_images)]

            # 获取文件名
            label_files = os.listdir(label_path)
            result_files = os.listdir(result_path)

            # 创建DataFrame保存结果
            results_df = pd.DataFrame({
                "Model": model_name,
                "Average Mode": average_mode,
                "Label File": label_files,
                "Result File": result_files,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "IoU (Jaccard Score)": iou
            })

            # 将当前结果添加到all_results列表中
            all_results.append(results_df)
        except:
            print(f"Error: {model_name} {average_mode}")

# 将所有结果合并为一个DataFrame
final_results_df = pd.concat(all_results, ignore_index=True)

# 将合并的结果写入一个Excel文件
final_results_df.to_excel("final_results_vote.xlsx", index=False)

print("All results saved to final_results.xlsx")
