import numpy as np
import pywt

class ImageTransform:
    def __init__(self, image, wavelet='db2', level=3):
        """图像变换类，用于执行小波变换和频域分块传输处理

        Args:
            image: 输入的图像，二维数组
            wavelet: 使用的小波名称，默认是'db2'
            level: 小波变换的层数，默认是3
        """             
        self.image = np.float32(image)
        self.wavelet = wavelet
        self.level = level
        self.coeffs = []
        self.LL = None      # 低频部分
        self.LH = None      # 水平高频部分
        self.HL = None      # 垂直高频部分
        self.HH = None      # 细节高频部分

    def wavelet_transform(self):
        """对图像进行多层小波变换，生成多层分解后的频域信息。

        Returns:
            coeffs: 变换后的频域矩阵， 大小为[level, block_size, block_size]
            block_size: 每个level中block_size的大小, 长度为level
        """        
        current_image = self.image
        block_size = []
        for i in range(self.level):
            coeff = pywt.dwt2(current_image, self.wavelet)
            LL, (LH, HL, HH) = coeff
            block_size.append(LL.shape)
            self.coeffs.append((LL, LH, HL, HH))
            # 为下一层的小波变换做准备
            current_image = LL
        print(f"Wavelet transform complete. Levels: {self.level}")
        return self.coeffs, block_size