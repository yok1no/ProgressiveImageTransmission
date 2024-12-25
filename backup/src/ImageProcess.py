import numpy as np
import pywt

class ImageTransform:
    def __init__(self, image, wavelet='db2', block_size=64):
        """
        图像变换类，用于执行小波变换和频域分块处理。
        
        :param image: 输入的图像，二维数组
        :param wavelet: 使用的小波名称，默认是'db2'
        :param block_size: 分块的大小，默认是64
        """
        self.image = np.float32(image)  # 确保图像是浮点型
        self.wavelet = wavelet
        self.block_size = block_size
        self.coeffs = None  # 存储小波变换系数
        self.LL = None      # 低频部分
        self.LH = None      # 水平高频部分
        self.HL = None      # 垂直高频部分
        self.HH = None      # 细节高频部分

    def wavelet_transform(self):
        """
        对图像执行二维离散小波变换，保存频域信息。
        """
        self.coeffs = pywt.dwt2(self.image, self.wavelet)
        self.LL, (self.LH, self.HL, self.HH) = self.coeffs
        print(f"Wavelet transform complete. LL shape: {self.LL.shape}")

    def split_frequency_blocks(self):
        """
        将频域信息（LL, LH, HL, HH）分块。
        
        :return: 一个字典，包含分块后的低频部分和三个高频部分
        """
        blocks = {
            "LL": self._split_into_blocks(self.LL),
            "LH": self._split_into_blocks(self.LH),
            "HL": self._split_into_blocks(self.HL),
            "HH": self._split_into_blocks(self.HH),
        }
        return blocks

    def _split_into_blocks(self, component):
        """
        将频域中的某一部分（如LL、LH等）分块，并返回一个二维的索引列表。

        :param component: 频域的某一部分（二维数组）
        :return: 分块后的二维块索引数组
        """
        blocks = []
        height, width = component.shape
        num_blocks_per_dim = height // self.block_size  # 计算每一维上有多少个块

        # 将每个块的索引以二维坐标 (row, col) 的方式保存
        for i in range(num_blocks_per_dim):
            row_blocks = []  # 每一行的块列表
            for j in range(num_blocks_per_dim):
                # 计算每个块的位置，并提取该块的数据
                row_start, row_end = i * self.block_size, (i + 1) * self.block_size
                col_start, col_end = j * self.block_size, (j + 1) * self.block_size
                block = component[row_start:row_end, col_start:col_end]
                row_blocks.append(block)
            blocks.append(row_blocks)  # 每一行的块加入 blocks
        return blocks
