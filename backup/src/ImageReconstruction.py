import numpy as np
import pywt  # 小波变换库，用于反变换
import matplotlib.pyplot as plt

class ImageReconstruction:
    def __init__(self, image_shape, wavelet="db2", block_size=64):
        """
        图像重建类，逐步重建图像。
        
        :param image_shape: 原始图像的形状 (height, width)
        :param wavelet: 小波基类型，默认为 'db2'
        :param block_size: 分块大小
        """
        self.image_shape = image_shape
        self.wavelet = wavelet
        self.block_size = block_size
        self.figure = None
        self.ax = None
        
        # 初始化频域数据的存储结构
        self.freq_blocks = {
            "LL": np.zeros((image_shape[0] // 2, image_shape[1] // 2), dtype=np.float32),
            "LH": np.zeros((image_shape[0] // 2, image_shape[1] // 2), dtype=np.float32),
            "HL": np.zeros((image_shape[0] // 2, image_shape[1] // 2), dtype=np.float32),
            "HH": np.zeros((image_shape[0] // 2, image_shape[1] // 2), dtype=np.float32),
        }
        # 标记每个块是否已收到
        self.block_received = {key: np.zeros((image_shape[0] // (2 * block_size), 
                                              image_shape[1] // (2 * block_size)), dtype=bool)
                               for key in self.freq_blocks}

    def add_received_block(self, block_type, block_index, block_data):
        block_size = self.block_size
        row, col = block_index
        row_start, row_end = row * block_size, (row + 1) * block_size
        col_start, col_end = col * block_size, (col + 1) * block_size
        
        self.freq_blocks[block_type][row_start:row_end, col_start:col_end] = block_data
        self.block_received[block_type][row, col] = True

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
        self.ax.clear()
        self.ax.imshow(reconstructed_image, cmap="gray")
        self.ax.set_title("Progressive Image Reconstruction")
        self.ax.axis("off")
        plt.draw()
        plt.pause(0.1)  # 控制更新速度

    def is_complete(self):
        """
        检查是否所有块都已接收。
        """
        return all(np.all(received) for received in self.block_received.values())

    def reconstruct_image(self):
        """
        从当前的频域数据重建空间域图像。
        
        :return: 重建后的图像 (numpy.ndarray)
        """
        coeffs2 = (
            self.freq_blocks["LL"],
            (self.freq_blocks["LH"], self.freq_blocks["HL"], self.freq_blocks["HH"]),
        )
        # 进行二维小波反变换
        reconstructed_image = pywt.idwt2(coeffs2, wavelet=self.wavelet)
        return np.clip(reconstructed_image, 0, 255).astype(np.uint8)
