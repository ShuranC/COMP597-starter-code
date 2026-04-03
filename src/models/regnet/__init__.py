from src.models.regnet.regnet import regnet_init
import src.config as config
import src.trainer as trainer

from typing import Any, Dict, Optional, Tuple
import torch.utils.data as data

model_name = "regnet"

def init_model(conf: config.Config, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    return regnet_init(conf, dataset)
