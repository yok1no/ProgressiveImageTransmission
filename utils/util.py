import numpy as np
import imageio
import tempfile

def encode_block(block, quality_mode="dB"):
    """使用 imageio 对数据块进行 JPEG2000 压缩。

    Args:
        block: 待压缩的数据块
        quality_mode: 压缩模式， 默认为"dB"

    Returns:
        compressed_data: 压缩后的数据块
        block_min: 块中最小的元素
        block_max: 块中最大的元素
    """    
    block_min, block_max = block.min(), block.max()
    normalized_block = ((block - block_min) / (block_max - block_min) * 65535).astype(np.uint16)

    with tempfile.NamedTemporaryFile(suffix=".jp2", delete=False) as temp_file:
        imageio.imwrite(temp_file.name, normalized_block, format="JP2", quality_mode=quality_mode)
        with open(temp_file.name, "rb") as f:
            compressed_data = f.read()

    return compressed_data, block_min, block_max

def decode_block(compressed_data, block_min, block_max):
    """解压JPEG2000压缩数据,并还原为float32类型

    Args:
        compressed_data: 压缩数据
        block_min: 块中最小的元素
        block_max: 块中最大的元素

    Returns:
        restored_block: 解压后的数据块
    """    
    with tempfile.NamedTemporaryFile(suffix=".jp2", delete=False) as temp_file:
        temp_file.write(compressed_data)
        temp_file.flush()
        decompressed_data = imageio.imread(temp_file.name, format="JP2")

    restored_block = decompressed_data.astype(np.float32) / 65535 * (block_max - block_min) + block_min
    return restored_block