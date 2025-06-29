import argparse
import logging
import os

from huggingface_hub import snapshot_download


logger = logging.getLogger(__name__)

def setup_logging(log_level="INFO", log_file="huggingface_downloader.log"):
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except IOError as e:
        root_logger.error(f"Could not open log file {log_file} for writing. {e}")


def download_model(repo_id: str, target_dir: str, revision: str = "main", ignore_patterns: list = []):
    logger.info(f"Starting download for model: {repo_id}")
    logger.info(f"Target directory: {target_dir}")
    logger.info(f"Revision: {revision}")
    if ignore_patterns:
        logger.info(f"Ignoring patterns: {ignore_patterns}")

    try:
        if not os.path.exists(target_dir):
            logger.info(f"Target directory '{target_dir}' does not exist. Creating it...")
            os.makedirs(target_dir)
            logger.info(f"Successfully created directory: {target_dir}")
    except OSError as e:
        logger.info(f"Error: Could not create directory '{target_dir}'. {e}")
        return

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            revision=revision,
            ignore_patterns=ignore_patterns,
        )
        logger.info("\nModel download completed successfully!")
        logger.info(f"Model files are saved in: {os.path.abspath(target_dir)}")
    except Exception as e:
        logger.info(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    DEFAULT_TARGET_DIR = "./downloaded_models"

    parser = argparse.ArgumentParser(
        description="Download a model from the Hugging Face Hub.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        help="The repository ID of the model to download."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default=DEFAULT_TARGET_DIR,
        help=f"The local directory to save the model to.\n(default: '{DEFAULT_TARGET_DIR}')"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The specific model revision (branch, tag, or commit hash).\n(default: 'main')"
    )
    parser.add_argument(
        "--ignore",
        nargs='*',
        help="Space-separated list of file patterns to ignore (e.g., '*.safetensors' '*.onnx')."
    )

    args = parser.parse_args()

    model_specific_dir = os.path.join(args.target_dir, args.repo_id.replace("/", "_"))

    download_model(
        repo_id=args.repo_id,
        target_dir=model_specific_dir,
        revision=args.revision,
        ignore_patterns=args.ignore
    )

