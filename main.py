from src.ImageProcess import ImageTransform
from src.Transmission import ProgressiveTransmission
from src.ImageReconstruction import ImageReconstruction
import cv2
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="parameter list of Progressive transmission system")
    parser.add_argument(
        "--encode_efficiency_dir",
        type=str,
        default="data/result/encode_efficiency.png",
        help="the address to save the encode efficiency image"
    )
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
        "--level",
        type=int,
        default=5,
        help="the level of wavelet"
    )
    parser.add_argument(
        "--quality",
        type=str,
        choices=["rates", "dB"], 
        default="dB",
        help="the quality of encode, choose between 'rates' or 'dB'"
    )
    parser.add_argument(
        "--band_width",
        type=int,
        default=16777216,
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
    transformer = ImageTransform(image, args.wavelet, args.level)
    # 执行小波变换
    coeffs, block_size = transformer.wavelet_transform()

    transmission = ProgressiveTransmission(coeffs, args.level, args.band_width, args.quality)
    reconstruction = ImageReconstruction(image_shape, block_size, args.level, args.wavelet)

    while True:
        encoded_block = transmission.transmit_next()
        if encoded_block is None:
            break
        level, block_type, restored_data = transmission.decode_received_data(encoded_block)
        reconstruction.add_received_block(level, block_type, restored_data)
    plt.pause(2)

    transmission.plot_efficiency(args.encode_efficiency_dir)
        


    
