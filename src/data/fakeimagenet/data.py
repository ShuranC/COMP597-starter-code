import datasets
import src.config as config
import torch
import torch.utils.data
from PIL import Image

data_load_name = "fakeimagenet"


def load_data(conf: config.Config) -> torch.utils.data.Dataset:
    """Generates a fake ImageNet-like dataset of random RGB images in memory.

    Follows the same approach as milabench's datagen.py: each image is a
    random uint8 tensor converted to a PIL Image, and labels are assigned
    as offset % num_classes so all classes are represented evenly.
    """
    fakenet_conf = conf.data_configs.fakeimagenet
    num_samples = fakenet_conf.num_samples
    num_classes = fakenet_conf.num_classes
    image_size = fakenet_conf.image_size
    seed = fakenet_conf.seed

    rng = torch.Generator()
    rng.manual_seed(seed)

    images = []
    labels = []
    for i in range(num_samples):
        pixel_data = torch.randint(
            0, 256,
            (3, image_size, image_size),
            dtype=torch.uint8,
            generator=rng,
        )
        img = Image.fromarray(pixel_data.permute(1, 2, 0).numpy(), mode="RGB")
        images.append(img)
        labels.append(i % num_classes)

    return datasets.Dataset.from_dict({"image": images, "label": labels})
