# pyright: reportMissingImports=false

import albumentations as A
from pathlib import Path
import os


# TODO: put this information into a config file
# IMG_SIZE = 224
IMG_PADDING = (15, 15, 15, 30)  # left, top, right, bottom

# font is in the same dir as the augmentation file
font_dir = Path((os.path.abspath(__file__))).parent
font_fname = "LiberationSerif-Regular.ttf"
font_path = font_dir / font_fname


adv_data_transform = A.Compose(
    [
        A.Normalize(mean=0.0, std=1.0),
        A.Pad(padding=IMG_PADDING, p=0.5),
        A.Resize(height=224, width=224),
        A.Rotate(limit=10, p=0.5),
        # A.CLAHE(clip_limit=2, tile_grid_size=(10, 10), p=0.3),
        A.TextImage(font_path=font_path, p=0.5, font_color="white"),
    ]
)

data_transform = A.Compose([A.Normalize(mean=0.0, std=1.0), ])
                            # A.Resize(height=IMG_SIZE, width=IMG_SIZE)])


AUGMENTATIONS = {"advanced": adv_data_transform, "standard": data_transform}


if __name__ == "__main__":
    print(font_path)
