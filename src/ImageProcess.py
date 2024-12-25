import numpy as np
import pywt

class ImageTransform:
    def __init__(self, image, wavelet='db2', level=1):
        """
        图像变换类，用于执行小波变换和频域大块传输处理。
        
        :param image: 输入的图像，二维数组
        :param wavelet: 使用的小波名称，默认是'db2'
        :param block_size: 分块的大小，默认是64
        :param level: 小波变换的层数，默认是1（1层小波分解）
        """
        self.image = np.float32(image)  # 确保图像是浮点型
        self.wavelet = wavelet
        self.level = level
        self.coeffs = []  # 存储小波变换系数
        self.LL = None      # 低频部分
        self.LH = None      # 水平高频部分
        self.HL = None      # 垂直高频部分
        self.HH = None      # 细节高频部分

    def wavelet_transform(self):
        """
        对图像进行多层小波变换，生成多层分解后的频域信息。
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