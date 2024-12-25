from src.ImageProcess import ImageTransform
from src.Transmission import ProgressiveTransmission
from src.ImageReconstruction import ImageReconstruction
import cv2
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="parameter list of Progressive transmission system")
    parser.add_argument(
        "--input_image",
        type=str,
        default="data/input/4.jpg",
        help="the address of input image"
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="db6",
        help="the type of wavelet"
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=64,
        help="the size of block"
    )
    parser.add_argument(
        "--band_width",
        type=int,
        default=16384,
        help="the size of block"
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # 读取图像
    image = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
    image_shape = image.shape
    print(f"image shape: {image_shape}")
    # 创建ImageTransform对象
    transformer = ImageTransform(image, args.wavelet, args.block_size)
    # 执行小波变换
    transformer.wavelet_transform()
    # 分块
    blocks = transformer.split_frequency_blocks()

    transmission = ProgressiveTransmission(blocks, args.band_width)
    reconstruction = ImageReconstruction(image_shape, args.wavelet, args.block_size)

    while True:
        encoded_block = transmission.transmit_next()
        if encoded_block is None:
            break
        block_type, block_index, block_data = transmission.decode_received_block(encoded_block)
        reconstruction.add_received_block(block_type, block_index, block_data)

    plt.pause(2) 
        


    
