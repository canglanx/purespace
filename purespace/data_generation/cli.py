import argparse
import logging
import os
import random
import sys

import numpy as np
import yaml

from purespace.data_generation.pipeline import run_data_generation


logger = logging.getLogger(__name__)


def setup_logging(log_dir: str, is_debug: bool = False) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "data_generation.log")
    log_level = logging.DEBUG if is_debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
    )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Purespace Data Generation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to generation configuration file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging"
    )
    args = parser.parse_args()

    # Setup logging
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(module_dir))
    setup_logging(os.path.join(project_dir, "logs"), args.debug)

    # Set random seed if provided
    if args.seed is not None:
        set_random_seed(args.seed)
        logger.info("Random seed is set to: %s", args.seed)

    # Load config file
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        logger.error("Configuration file not found at: %s", config_path)
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except Exception as e:
        logger.error("Failed to parse YAML configuration:\n%s", e)
        sys.exit(1)

    # Run data generation
    logger.info("Start data generation using config: %s", config_path)
    try:
        run_data_generation(config_dict)
        logger.info("Data generation finished")
    except Exception:
        logger.exception("Data generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
