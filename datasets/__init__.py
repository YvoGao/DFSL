
from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc_aircraft import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenet import ImageNet
from .imagenetv2 import ImageNetV2
from .imagenet_sketch import ImageNetSketch
from .imagenet_r import ImageNetR
from .imagenet_a import ImageNetA
from .cub import CUB200


dataset_list = {
                "OxfordPets": OxfordPets,
                "EuroSAT": EuroSAT,
                "UCF101": UCF101,
                "SUN397": SUN397,
                "Caltech101": Caltech101,
                "DescribableTextures": DescribableTextures,
                "FGVCAircraft": FGVCAircraft,
                "Food101": Food101,
                "OxfordFlowers": OxfordFlowers,
                "StanfordCars": StanfordCars,
                "ImageNet": ImageNet,
                "ImageNetV2": ImageNetV2,
                "ImageNetSketch": ImageNetSketch,
                "ImageNetR": ImageNetR,
                "ImageNetA": ImageNetA,
                "CUB200": CUB200
                }


def build_dataset(cfg):
        return dataset_list[cfg.dataset](cfg)

