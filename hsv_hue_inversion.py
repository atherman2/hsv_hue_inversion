import numpy as np
import cv2
import argparse
from typing import Callable
from os import path

def hsv_image_process(pixel):
    pixel = pixel.astype(np.float32)
    pixel *= 2
    pixel = angle_numpy(pixel - 180)
    pixel /= 2
    return pixel.astype(np.uint8)

def angle(x: float) -> float:
    while x >= 360.0:
        x -= 360.0
    while x < 0.0:
        x += 360.0
    return x

def angle_numpy(x):
    return np.mod(x, 360.0)

def interval(h: float, d: float) -> Callable[[np.ndarray], np.ndarray]:
    begin = angle(h - d)
    end = angle(h + d)
    def inner(x: np.ndarray) -> np.ndarray:
        nonlocal begin, end
        x = angle_numpy(x)
        cond1 = (begin <= end) & (begin <= x) & (x <= end)
        cond2 = (end < begin) & ((begin <= x) | (end >= x))
        return cond1 | cond2
    return inner

def invert_angle(x):
    return angle(x - 180.0)

def parse_arg_data() -> tuple[str, float, float, bool]:
    args = create_parser().parse_args()
    return (args.image_file_path[0], float(args.hue_value[0]), float(args.hue_amplitude[0]), args.save)

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_file_path", nargs=1, help="Image file path")
    parser.add_argument("hue_value", nargs=1, help="Central value of hue range that will be inverted")
    parser.add_argument("hue_amplitude", nargs=1, help="""Amplitude of hue range that will be inverted.
The hue range starts at (hue_value - hue_amplitude) and ends at (hue_value + amplitude)""")
    parser.add_argument("-s", "--save", action="store_true", help="""Optional flag to enable output image file saving.
The output image file will be saved as \"out_<path>\"""")
    return parser

def main() -> None:
    try:
        image_path, h, d, save = parse_arg_data()
        image = cv2.imread(path.abspath(image_path), 1)
    except Exception:
        print("""Unable to parse arguments. It is necessary to provide: <image_file_path> <hue_value> <hue_amplitude>.
It is necessary that <image_file_path> be a image file path, and <hue_value> and <hue_amplitude> be numbers""")
        return
    is_in_interval = interval(h, d)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels_in_interval = (lambda pixel: is_in_interval(2 * pixel))(hsv_image[:, :, 0].astype(np.float32))
    hsv_image[:, :, 0] = np.where(pixels_in_interval, hsv_image_process(hsv_image[:, :, 0]), hsv_image[:, :, 0])
    show_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    cv2.imshow("Hue inverted image according to hue range", show_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
