import torch


MODEL_NAMES = ["bigscience/bloom-petals", "bigscience/bloomz-petals", "bigscience/bloom-7b1-petals"]
DEFAULT_MODEL_NAME = "bigscience/bloom-petals"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16

STEP_TIMEOUT = 5 * 60
MAX_SESSIONS = 50  # Has effect only for API v1 (HTTP-based)
