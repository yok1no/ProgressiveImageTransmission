import os
import numpy as np
import pywt  # 小波变换库，用于反变换
import matplotlib.pyplot as plt

class ImageReconstruction:
    def __init__(self, origin_image, block_size, level = 3, wavelet="db2"):
        """
        图像重建类，逐步重建图像。
        
        :param image_shape: 原始图像的形状 (height, width)
        :param wavelet: 小波基类型，默认为 'db2'
        :param block_size: 分块大小
        """
        self.origin_image = np.float32(origin_image)
        self.image_shape = origin_image.shape
        self.wavelet = wavelet
        self.level = level
        self.block_size = block_size
        #init coeffs to zero due to block_size
        self.coeffs = [np.zeros((4 * block_size[i][0], 4 * block_size[i][1])) for i in range(level)]
        self.figure = None
        self.ax = None
        self.mse_losses = []
        

    def add_received_block(self, level, block_type, block_data):
        block_size = self.block_size[level][0]
        if block_type == "LL":
            row_start, row_end = 0, block_size
            col_start, col_end = 0, block_size
        elif block_type == "LH":
            row_start, row_end = 0, block_size
            col_start, col_end = block_size, 2 * block_size
        elif block_type == "HL":
            row_start, row_end = block_size, 2 * block_size
            col_start, col_end = 0, block_size
        elif block_type == "HH":
            row_start, row_end = block_size, 2 * block_size
            col_start, col_end = block_size, 2 * block_size
        
        self.coeffs[level][row_start:row_end, col_start:col_end] = block_data

        # 更新显示
        self._update_display()

    def _update_display(self):
        """
        更新显示当前阶段的图像重建结果。
        """
        if self.figure is None or self.ax is None:
            # 初始化绘图窗口
            self.figure, self.ax = plt.subplots()
            plt.ion()  # 打开交互模式

        # 重建当前图像
        reconstructed_image = self.reconstruct_image()

        mse = self.calculate_mse(self.origin_image, reconstructed_image)
        self.mse_losses.append(mse)  # 保存 MSE 损失

        self.ax.clear()
        self.ax.imshow(reconstructed_image, cmap="gray")
        self.ax.set_title("Progressive Image Reconstruction")
        self.ax.axis("off")
        plt.draw()
        plt.pause(0.1)  # 控制更新速度

    def reconstruct_image(self):
        """
        使用小波逆变换从小波系数逐层重建图像。
        从最后一层开始，逐步恢复出原始图像。
        """
        # 从最后一层开始逐步重建
        reconstructed_image = None

        for level_idx in reversed(range(self.level)):
            # 当前层的小波系数 (LL, LH, HL, HH)
            coeffs_level = self.coeffs[level_idx]
            
            # 切分各个方向的系数块
            LL = coeffs_level[0:self.block_size[level_idx][0], 0:self.block_size[level_idx][1]]
            LH = coeffs_level[0:self.block_size[level_idx][0], self.block_size[level_idx][1]:2*self.block_size[level_idx][1]]
            HL = coeffs_level[self.block_size[level_idx][0]:2*self.block_size[level_idx][0], 0:self.block_size[level_idx][1]]
            HH = coeffs_level[self.block_size[level_idx][0]:2*self.block_size[level_idx][0], self.block_size[level_idx][1]:2*self.block_size[level_idx][1]]

            # 合并成一个系数元组
            coeff_tuple = (LL, (LH, HL, HH))
            
            # 如果是最后一层，直接使用当前系数块重建图像
            if reconstructed_image is None:
                reconstructed_image = pywt.idwt2(coeff_tuple, wavelet=self.wavelet)
            else:
                # 否则，进行上采样并重建
                reconstructed_image = pywt.idwt2((reconstructed_image, coeff_tuple[1]), wavelet=self.wavelet)
            reconstructed_image = self.crop_to_expected(reconstructed_image, level_idx - 1)

        # 最终重建的图像已经恢复为原始尺寸
        return reconstructed_image
    
    def crop_to_expected(self, image, level):
        if level == -1:
            return image
        else:
            return image[0:self.block_size[level][0], 0:self.block_size[level][1]]
    
    def calculate_mse(self, original_image, reconstructed_image):
        """
        计算原图与重建图像之间的均方误差（MSE）损失。
        """
        return np.mean((original_image - reconstructed_image) ** 2)
    
    def plot_loss(self, mes_losses_dir):
        """
        绘制 MSE 损失的折线图，并保存。
        """
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.mse_losses, marker='o', linestyle='-', color='b', label="MSE Loss")
        plt.xlabel("Reconstruction Step")
        plt.ylabel("MSE Loss")
        plt.title("MSE Loss Curve during Image Reconstruction")
        plt.grid(True)
        
        mes_losses_dir = os.path.join(mes_losses_dir, "loss_curve.png")
        plt.savefig(mes_losses_dir)
        plt.close()  # 关闭当前图像，避免图像累积
