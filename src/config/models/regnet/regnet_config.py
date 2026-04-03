from src.config.util.base_config import _Arg, _BaseConfig

class ModelConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_num_classes = _Arg(type=int, help="Number of output classes.", default=1000)
