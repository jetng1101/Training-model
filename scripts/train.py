import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.data import build_dataloaders
from src.models import Cifar100Classifier
from src.training import Trainer
from src.utils import Logger, set_seed

load_dotenv()


@hydra.main(config_path="../configs", config_name="training", version_base="1.3")
def main(config: DictConfig) -> None:
    set_seed(config.seed)

    dataloaders = build_dataloaders(config.data)
    model = Cifar100Classifier(config.model)
    logger = Logger(config)

    trainer = Trainer(
        model=model,
        train_loader=dataloaders.train,
        val_loader=dataloaders.val,
        config=config,
        logger=logger,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
