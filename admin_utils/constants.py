"""
Useful constant variables.
"""

from pathlib import Path

# pylint: disable=invalid-name,too-few-public-methods
try:
    import torch
except ImportError:
    print('Library "torch" not installed. Failed to import.')

    class torch:
        """
        Mock class for public repo that does not have requirements.
        """

        class cuda:
            """
            Mock class for torch.cuda.
            """

            @staticmethod
            def is_available() -> bool:
                """
                Mock method for torch.cuda.is_available.
                """
                return False


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Selected {DEVICE} for all reference collection tasks")

PROJECT_ROOT = Path(__file__).parent.parent
PROJECT_CONFIG_PATH = PROJECT_ROOT / "project_config.json"
CONFIG_PACKAGE_PATH = PROJECT_ROOT / "config"
TRACKED_JSON_PATH = (
    PROJECT_ROOT / "admin_utils" / "external_pr_files" / "tracked_files.json"
).relative_to(PROJECT_ROOT)

DIST_PATH = PROJECT_ROOT / "dist"

GLOBAL_SEED = 77
GLOBAL_NUM_SAMPLES = 100
GLOBAL_MAX_LENGTH = 120
GLOBAL_INFERENCE_BATCH_SIZE = 3
GLOBAL_FINE_TUNING_BATCH_SIZE = 3
