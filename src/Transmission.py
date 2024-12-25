import numpy as np
from utils.util import encode_block, decode_block

class ProgressiveTransmission:
    def __init__(self, coeffs, level, bandwidth=16777216):
        """
        渐进传输类，支持编码与纠错。
        
        :param blocks: 包含分块后的频域数据的字典 { "LL": [...], "LH": [...], "HL": [...], "HH": [...] }
        :param bandwidth: 每次传输的最大数据量（字节），默认 4096
        """
        self.coeffs = coeffs
        self.level = level
        self.bandwidth = bandwidth
        self.transmission_queue = self._create_transmission_queue()

    def _create_transmission_queue(self):
        """
        创建传输队列，按渐进式顺序（从最细节到低频）进行排序。
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
        """
        对频域数据进行编码（可以使用JPEG2000或其他方式）。
        
        :param data: 频域数据（LL、LH、HL、HH）
        :return: 编码后的数据块
        """
        compressed_data, block_min, block_max = encode_block(data, quality=50)
        return compressed_data, block_min, block_max

    def transmit_next(self):
        """
        模拟传输下一个频域数据块。
        
        :return: 当前传输的数据块信息，包括 (block_type, level, compressed_data)
        """
        if not self.transmission_queue:
            print("All frequency domain data has been transmitted.")
            return None

        # 获取队列中的下一个数据块
        block_type, level, data = self.transmission_queue.pop(0)
        compressed_data, block_min, block_max = self.encode_frequency_data(data)
        block_size = len(compressed_data)
        
        if block_size > self.bandwidth:
            raise ValueError(f"Block size ({block_size} bytes) exceeds bandwidth ({self.bandwidth} bytes).")
        
        print(f"Transmitting {block_type} block from level {level}, original size: {data.shape}, encoded size: {block_size} bytes.")
        return block_type, level, compressed_data, block_min, block_max

    def decode_received_data(self, encoded_data):
        """
        解码接收到的频域数据（LL、LH、HL、HH）。
        
        :param encoded_data: 包含 (block_type, level, compressed_data, block_min, block_max)
        :return: 解码后的频域数据
        """
        block_type, level, compressed_data, block_min, block_max = encoded_data
        restored_data = decode_block(compressed_data, block_min, block_max)
        return level, block_type, restored_data
