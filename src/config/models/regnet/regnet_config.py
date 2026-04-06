from src.config.util.base_config import _Arg, _BaseConfig

class ModelConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_num_classes = _Arg(type=int, help="Number of output classes.", default=1000)
        self._arg_max_duration_seconds = _Arg(type=float, help="Maximum training duration in seconds. 0 means run until dataset is exhausted.", default=0)
