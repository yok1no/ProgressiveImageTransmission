import os
from utils.util import encode_block, decode_block
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'SimHei'  # SimHei 是黑体，你也可以使用其他字体，如 Microsoft YaHei
rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

class ProgressiveTransmission:
    def __init__(self, coeffs, level, bandwidth=16777216, quality = "dB"):
        """渐进传输类，支持编码与纠错

        Args:
            coeffs: 包含分块后的频域数据的列表 [(LL, LH, HL, HH), (LL, LH, HL, HH),...]
            level: 小波变换的层数
            bandwidth: 每次传输的最大数据量（字节），默认 16777216
            quality: 编码方式， dB 或者 rates
        """        
        self.coeffs = coeffs
        self.level = level
        self.bandwidth = bandwidth
        self.transmission_queue = self._create_transmission_queue()
        self.efficiency_list = []
        self.quality = quality

    def _create_transmission_queue(self):
        """创建传输队列，按渐进式顺序（从最细节到低频）进行排序

        Returns:
            queue: 传输队列
        """        
        queue = []
        
        # 逆序遍历各层的频域信息
        for level in range(self.level - 1, -1, -1):
            LL, LH, HL, HH = self.coeffs[level]
            # 每一层的细节优先传输（从最细节到低频）
            queue.append(('LL', level, LL))
            queue.append(('LH', level, LH))
            queue.append(('HL', level, HL))
            queue.append(('HH', level, HH))
        
        return queue

    def encode_frequency_data(self, data):
        """使用JPEG2000对频域数据进行编码

        Args:
            data: 频域数据

        Returns:
            compressed_data: 编码后的数据块
            block_min: 块中最小的元素
            block_max: 块中最大的元素
            original_size: 原始数据块的大小
            compressed_size: 编码后的数据块的大小
        """        

        compressed_data, block_min, block_max = encode_block(data, self.quality)
        original_size = data.nbytes
        compressed_size = len(compressed_data)
        return compressed_data, block_min, block_max, original_size, compressed_size

    def transmit_next(self):
        """模拟传输频域数据块

        Raises:
            ValueError: 当数据块大小超过带宽时报错

        Returns:
            block_type: 数据块类型
            level: 数据块所在的层级
            compressed_data: 编码后的数据块
            block_min: 块中最小的元素
            block_max: 块中最大的元素
        """        
        if not self.transmission_queue:
            print("All frequency domain data has been transmitted.")
            return None

        # 获取队列中的下一个数据块
        block_type, level, data = self.transmission_queue.pop(0)
        compressed_data, block_min, block_max, original_size, compressed_size = self.encode_frequency_data(data)
        block_size = len(compressed_data)
        
        efficiency = compressed_size / original_size
        self.efficiency_list.append(efficiency)

        if block_size > self.bandwidth:
            raise ValueError(f"Block size ({block_size} bytes) exceeds bandwidth ({self.bandwidth} bytes).")
        
        print(f"Transmitting {block_type} block from level {level}, original size: {data.shape}, "
              f"encoded size: {block_size} bytes, efficiency: {efficiency:.4f}.")
        return block_type, level, compressed_data, block_min, block_max

    def decode_received_data(self, encoded_data):
        """解码接收到的频域数据

        Args:
            encoded_data: (block_type, level, compressed_data, block_min, block_max)

        Returns:
            level: 数据块所在的层级
            block_type: 数据块类型
            restored_data: 解码后的频域数据
        """        
        block_type, level, compressed_data, block_min, block_max = encoded_data
        restored_data = decode_block(compressed_data, block_min, block_max)
        return level, block_type, restored_data

    def plot_efficiency(self, encode_efficiency_dir):
        """绘制编码效率的折线图，并保存

        Args:
            encode_efficiency_dir: 保存编码效率图像的地址
        """        
        if not self.efficiency_list:
            print("No efficiency data to plot.")
            return

        # 绘制编码效率的折线图
        plt.figure(figsize=(10, 6))
        plt.plot(self.efficiency_list, marker='o', linestyle='-', color='b', label="压缩率")
        plt.xlabel("传输块")
        plt.ylabel("压缩率")
        plt.title("压缩率")
        plt.grid(True)
        
        # 计算平均编码效率
        average_efficiency = sum(self.efficiency_list) / len(self.efficiency_list)
        plt.axhline(y=average_efficiency, color='r', linestyle='--', label=f"平均压缩率: {average_efficiency:.4f}")
        plt.legend()
        
        encode_efficiency_dir = os.path.join(encode_efficiency_dir, "coding_efficiency.jpg")
        # 保存图像
        plt.savefig(encode_efficiency_dir)
        plt.close()