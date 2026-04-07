# === import necessary modules ===
import src.config as config  # Configurations
import src.trainer as trainer  # Trainer base class
import src.trainer.stats as trainer_stats  # Trainer statistics module

# === import necessary external modules ===
from typing import Any, Dict, Optional, Tuple
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import transformers

"""
This file contains the code to train a RegNet Y 128GF model using the Simple trainer.
It uses the RegNet implementation from TorchVision.
https://pytorch.org/vision/stable/models/regnet.html
"""


class RegNetTrainer(trainer.SimpleTrainer):
    """SimpleTrainer subclass for RegNet image classification.

    Overrides forward to call the model with image tensors and compute
    cross-entropy loss against class labels.
    """

    def __init__(self, max_duration_seconds: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.max_duration_seconds = max_duration_seconds

    def forward(self, i: int, batch: Any, model_kwargs: Dict[str, Any]) -> torch.Tensor:
        self.optimizer.zero_grad()
        outputs = self.model(batch["image"])
        return nn.functional.cross_entropy(outputs, batch["label"])

    def step(self, i: int, batch: Any, model_kwargs: Dict[str, Any]) -> torch.Tensor:
        if model_kwargs is None:
            model_kwargs = {}
        batch = self.process_batch(i, batch)

        sync = torch.cuda.is_available()

        if sync: torch.cuda.synchronize()
        self.stats.start_forward()
        loss = self.forward(i, batch, model_kwargs)
        if sync: torch.cuda.synchronize()
        self.stats.stop_forward()

        if sync: torch.cuda.synchronize()
        self.stats.start_backward()
        self.backward(i, loss)
        if sync: torch.cuda.synchronize()
        self.stats.stop_backward()

        if sync: torch.cuda.synchronize()
        self.stats.start_optimizer_step()
        self.optimizer_step(i)
        if sync: torch.cuda.synchronize()
        self.stats.stop_optimizer_step()

        return loss, None

    def train(self, model_kwargs: Optional[Dict[str, Any]]) -> None:
        import tqdm.auto
        if self.max_duration_seconds is None:
            return super().train(model_kwargs)

        if model_kwargs is None:
            model_kwargs = {}

        progress_bar = tqdm.auto.tqdm(desc="loss: N/A")
        deadline = time.monotonic() + self.max_duration_seconds

        self.stats.start_train()
        i = 0
        while time.monotonic() < deadline:
            for batch in self.loader:
                if time.monotonic() >= deadline:
                    break
                self.stats.start_step()
                loss, descr = self.step(i, batch, model_kwargs)
                self.stats.stop_step()
                self.stats.log_loss(loss)
                self.stats.log_step()
                progress_bar.update(1)
                i += 1
        self.stats.stop_train()
        progress_bar.close()
        self.stats.log_stats()


def process_dataset(dataset: data.Dataset) -> data.Dataset:
    """Applies ImageNet-standard transforms to the dataset.

    Args:
        dataset (data.Dataset): Raw HuggingFace image dataset with "image" and "label" columns.
    Returns:
        data.Dataset: Dataset with transforms applied lazily on access.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def apply_transform(examples):
        examples["image"] = [transform(img.convert("RGB")) for img in examples["image"]]
        return examples

    return dataset.with_transform(apply_transform)


def simple_trainer(conf: config.Config, model: nn.Module, dataset: data.Dataset, device: torch.device) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """Builds a RegNetTrainer for the simple training loop.

    Args:
        conf (config.Config): The configuration object.
        model (nn.Module): The RegNet model.
        dataset (data.Dataset): The transformed dataset.
        device (torch.device): The device to train on.
    Returns:
        Tuple[trainer.Trainer, Optional[Dict]]: The trainer and optional extra kwargs.
    """
    loader = data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=conf.learning_rate)
    scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(loader),
    )

    return RegNetTrainer(
        loader=loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=device,
        stats=trainer_stats.init_from_conf(conf=conf, device=device, num_train_steps=len(loader)),
        max_duration_seconds=conf.model_configs.regnet.max_duration_seconds if conf.model_configs.regnet.max_duration_seconds > 0 else None,
    ), None


def regnet_init(conf: config.Config, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    """Initializes RegNet Y 128GF and returns the appropriate trainer.

    Args:
        conf (config.Config): The configuration object.
        dataset (data.Dataset): The dataset to use for training.
    Returns:
        Tuple[trainer.Trainer, Optional[Dict]]: The initialized trainer and optional extra kwargs.
    """
    dataset = process_dataset(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.regnet_y_128gf(weights=None, num_classes=conf.model_configs.regnet.num_classes).to(device)

    if conf.trainer == "simple":
        return simple_trainer(conf, model, dataset, device)
    else:
        raise Exception(f"Unknown trainer type {conf.trainer}")
