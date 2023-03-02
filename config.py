import torch

from petals.constants import PUBLIC_INITIAL_PEERS


INITIAL_PEERS = PUBLIC_INITIAL_PEERS
# Set this to a list of multiaddrs to connect to a private swarm instead of the public one, for example:
# INITIAL_PEERS = ['/ip4/10.1.2.3/tcp/31234/p2p/QmcXhze98AcgGQDDYna23s4Jho96n8wkwLJv78vxtFNq44']

MODEL_NAMES = ["bigscience/bloom-petals", "bigscience/bloomz-petals", "bigscience/bloom-7b1-petals"]
DEFAULT_MODEL_NAME = "bigscience/bloom-petals"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16

STEP_TIMEOUT = 5 * 60
MAX_SESSIONS = 50  # Has effect only for API v1 (HTTP-based)
