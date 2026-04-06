from src.config.util.base_config import _Arg, _BaseConfig

class DataConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_num_samples = _Arg(type=int, help="Number of fake images to generate.", default=1000)
        self._arg_num_classes = _Arg(type=int, help="Number of ImageNet classes to use.", default=1000)
        self._arg_image_size = _Arg(type=int, help="Height and width of generated images.", default=256)
        self._arg_seed = _Arg(type=int, help="Random seed for reproducibility.", default=42)
