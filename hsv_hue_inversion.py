import numpy as np
import cv2
import argparse
from typing import Callable

def angle(x: float) -> float:
    while x >= 360.0:
        x -= 360.0
    while x < 0.0:
        x += 360.0
    return x

def interval(h: float, d: float) -> Callable[[float], bool]:
    begin = angle(h - d)
    end = angle(h + d)
    def inner(x: float) -> bool:
        nonlocal begin, end
        x = angle(x)
        return ((begin <= end) and (begin <= x <= end)) or ((end < begin) and ((begin <= x) or (end >= x)))
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
        image = cv2.imread(image_path, 1)
    except Exception:
        print("""Unable to parse arguments. It is necessary to provide: <image_file_path> <hue_value> <hue_amplitude>.
It is necessary that <image_file_path> be a image file path, and <hue_value> and <hue_amplitude> be numbers""")
        return

if __name__ == "__main__":
    main()
