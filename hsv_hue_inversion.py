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
    return x % 360.0

def angle_numpy(x):
    return np.mod(x, 360.0)

def interval(h: float, d: float) -> Callable[[np.ndarray], np.ndarray]:
    begin = angle(h - d)
    end = angle(h + d)
    def inner(x: np.ndarray) -> np.ndarray:
        nonlocal begin, end
        x = angle_numpy(x)
        cond1 = (begin <= end) & (begin <= x) & (x <= end)
        cond2 = (end < begin) & ((begin <= x) | (x <= end))
        return cond1 | cond2
    return inner

def process_image(image_path: str, h: float, d: float) -> np.ndarray:
    image = cv2.imread(path.abspath(image_path), 1)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    is_in_interval = interval(h, d)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # OpenCV H channel: [0,179] → scale to [0,359] for computation
    pixels_in_interval = is_in_interval(hsv_image[:, :, 0].astype(np.float32) * 2)
    hsv_image[:, :, 0] = np.where(
        pixels_in_interval,
        hsv_image_process(hsv_image[:, :, 0]),
        hsv_image[:, :, 0]
    )
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def parse_arg_data():
    args = create_parser().parse_args()
    return (args.image_file_path[0], float(args.hue_value[0]), float(args.hue_amplitude[0]), args.save)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_file_path", nargs=1, help="Image file path")
    parser.add_argument("hue_value", nargs=1, help="Central value of hue range (0-360)")
    parser.add_argument("hue_amplitude", nargs=1, help="Amplitude of hue range (0-180)")
    parser.add_argument("-s", "--save", action="store_true", help="Save output image as out_<filename>")
    return parser

def main() -> None:
    try:
        image_path, h, d, save = parse_arg_data()
        if not (0 <= h <= 360):
            print("Erro: H deve estar entre 0 e 360.")
            return
        if not (0 <= d <= 180):
            print("Erro: d deve estar entre 0 e 180.")
            return
    except Exception:
        print("Uso: python hsv_hur_inversion.py <imagem> <H> <d> [-s]")
        return

    result = process_image(image_path, h, d)

    if save:
        out_path = "out_" + path.basename(image_path)
        cv2.imwrite(out_path, result)
        print(f"Imagem salva como: {out_path}")

    cv2.imshow("Hue inverted image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()