import numpy as np
from utils.util import encode_block, decode_block
class ProgressiveTransmission:
    def __init__(self, blocks, bandwidth=4096):
        """
        渐进传输类，支持编码与纠错。
        
        :param blocks: 包含分块后的频域数据的字典 { "LL": [...], "LH": [...], "HL": [...], "HH": [...] }
        :param bandwidth: 每次传输的最大数据量（字节），默认 4096
        """
        self.blocks = blocks
        self.bandwidth = bandwidth
        self.transmission_queue = self._create_transmission_queue()

    def _create_transmission_queue(self):
        """
        根据优先级生成传输队列，并对数据块进行 JPEG2000 压缩。
        现在，每个块会传递二维索引 (row, col)。
        """
        queue = []
        
        for key in ["HH", "HL", "LH", "LL"]:
        # for key in ["LL", "LH", "HL", "HH"]:
            num_blocks_per_dim = len(self.blocks[key])
            for row in range(num_blocks_per_dim):
                for col in range(num_blocks_per_dim):
                    block = self.blocks[key][row][col]
                    compressed_block, block_min, block_max = encode_block(block, quality=50)
                    
                    # 将二维的 (row, col) 坐标添加到队列
                    queue.append((key, (row, col), compressed_block, block_min, block_max))
                    
        return queue

    def transmit_next(self):
        """
        模拟传输下一个数据块。
        
        :return: 当前传输的数据块信息，包括 (block_type, block_index, compressed_data)
        """
        if not self.transmission_queue:
            print("All blocks have been transmitted.")
            return None

        # 获取队列中的下一个压缩块
        block_type, block_index, compressed_block, block_min, block_max = self.transmission_queue.pop(0)
        block_size = len(compressed_block)
        
        if block_size > self.bandwidth:
            raise ValueError(f"Block size ({block_size} bytes) exceeds bandwidth ({self.bandwidth} bytes).")
        
        print(f"Transmitting {block_type} block {block_index}, size: {block_size} bytes.")
        return block_type, block_index, compressed_block, block_min, block_max

    def decode_received_block(self, encoded_block):
        """
        解码传输的 JPEG2000 数据块。
        
        :param encoded_block: 包含 (block_type, block_index, compressed_data, block_min, block_max)
        :return: 解码后的数据块
        """
        block_type, block_index, compressed_block, block_min, block_max = encoded_block
        restored_block = decode_block(compressed_block, block_min, block_max)
        return block_type, block_index, restored_block

