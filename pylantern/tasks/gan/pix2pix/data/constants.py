from enum import Enum


class DatasetType(str, Enum):
    IMAGE_MASK_FOLDERS = "image_mask_folders"
    FFHQ = "ffhq"
