import os
import cv2
import numpy as np

# 输入和输出目录
input_dir = 'input'
output_dir = 'output'

# 如果输出目录不存在，则创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取所有输入文件
input_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

# 锐化滤镜
sharpen_kernel = np.array([[0, -1, 0],
                            [-1, 5,-1],
                            [0, -1, 0]])

# 遍历所有输入文件
for file_name in input_files:
    # 枋建输入图像的完整路径
    input_path = os.path.join(input_dir, file_name)

    # 读取灰度图像（以灰度模式读取）
    original_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # 检查是否成功读取图像
    if original_image is None:
        print(f"无法读取图像: {input_path}")
        continue

    # 调整图像分辨率到 4096×4096，使用更高质量的插值方法
    target_resolution = (4096, 4096)
    resized_image = cv2.resize(original_image, target_resolution, interpolation=cv2.INTER_CUBIC)

    # 应用锐化操作
    sharpened_image = cv2.filter2D(resized_image, -1, sharpen_kernel)

    # 归一化到 [0, 1] 范围，再扩展到 12 位色深（0-4095）
    normalized_image = sharpened_image.astype(np.float32) / 255.0  # 归一化到 [0, 1]
    image_12bit = np.clip(normalized_image * 4095, 0, 4095).astype(np.uint16)

    # 增强对比度（可选） - 使用简单的线性对比度拉伸
    # 将 12 位图像的对比度增强，确保亮度范围适当
    min_val, max_val = np.min(image_12bit), np.max(image_12bit)
    contrast_enhanced = ((image_12bit - min_val) / (max_val - min_val) * 4095).astype(np.uint16)

    # 构建输出图像的路径
    output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_4096_12bit.png")

    # 保存处理后的图像
    cv2.imwrite(output_path, contrast_enhanced)

    print(f"图像 {file_name} 处理完成，已保存为 {output_path}")
