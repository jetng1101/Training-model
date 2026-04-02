from pathlib import Path
from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ServingConfig(BaseSettings):
    """Configuration for the grading inference service."""

    model_config_path: str = str(PROJECT_ROOT / "configs" / "model" / "model.yaml")
    checkpoint_path: str = str(PROJECT_ROOT / "checkpoints" / "best.pth")
    image_size: int = 32
    normalize_mean: list[float] = [0.5071, 0.4867, 0.4408]
    normalize_std: list[float] = [0.2675, 0.2565, 0.2761]
    class_labels: list[str] = [
        "apple", "aquarium_fish", "baby", "bear", "beaver",
        "bed", "bee", "beetle", "bicycle", "bottle",
        "bowl", "boy", "bridge", "bus", "butterfly",
        "camel", "can", "castle", "caterpillar", "cattle",
        "chair", "chimpanzee", "clock", "cloud", "cockroach",
        "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox",
        "girl", "hamster", "house", "kangaroo", "keyboard",
        "lamp", "lawn_mower", "leopard", "lion", "lizard",
        "lobster", "man", "maple_tree", "motorcycle", "mountain",
        "mouse", "mushroom", "oak_tree", "orange", "orchid",
        "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
        "plain", "plate", "poppy", "porcupine", "possum",
        "rabbit", "raccoon", "ray", "road", "rocket",
        "rose", "sea", "seal", "shark", "shrew",
        "skunk", "skyscraper", "snail", "snake", "spider",
        "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
        "tank", "telephone", "television", "tiger", "tractor",
        "train", "trout", "tulip", "turtle", "wardrobe",
        "whale", "willow_tree", "wolf", "woman", "worm",
    ]
    device: str = "cpu"
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "GRADING_"}


settings = ServingConfig()
