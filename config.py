import torch
from petals.constants import PUBLIC_INITIAL_PEERS

from data_structures import ModelInfo

MODELS = {
    "Llama 2": [
        ModelInfo(
            name="Stable Beluga 2 (70B)",
            model_card="https://huggingface.co/stabilityai/StableBeluga2",
            license="https://huggingface.co/stabilityai/StableBeluga2/blob/main/LICENSE.txt",
            repository="petals-team/StableBeluga2",
            aliases=["stabilityai/StableBeluga2"],
        ),
        ModelInfo(
            name="Llama 2 (70B-Chat)",
            model_card="https://huggingface.co/meta-llama/Llama-2-70b-chat-hf",
            license="https://bit.ly/llama2-license",
            repository="meta-llama/Llama-2-70b-chat-hf",
        ),
    ],
    "Falcon": [
        ModelInfo(
            name="Falcon 180B-Chat",
            model_card="https://huggingface.co/tiiuae/falcon-180B-chat",
            license="https://huggingface.co/spaces/tiiuae/falcon-180b-license/blob/main/LICENSE.txt",
            repository="tiiuae/falcon-180B-chat",
            public_api=False,
        ),
    ],
    "Llama": [
        ModelInfo(
            name="Guanaco-65B",
            model_card="https://huggingface.co/timdettmers/guanaco-65b",
            license="https://huggingface.co/timdettmers/guanaco-65b",
            repository="huggyllama/llama-65b",
            adapter="timdettmers/guanaco-65b",
        ),
        ModelInfo(
            name="Llama-65B",
            model_card="https://github.com/facebookresearch/llama/blob/llama_v1/MODEL_CARD.md",
            license="https://bit.ly/llama-license",
            repository="huggyllama/llama-65b",
        ),
    ],
    "BLOOM": [
        ModelInfo(
            name="BLOOMZ-176B",
            model_card="https://huggingface.co/bigscience/bloomz",
            license="https://bit.ly/bloom-license",
            repository="bigscience/bloomz",
        ),
    ],
}

INITIAL_PEERS = PUBLIC_INITIAL_PEERS
# Set this to a list of multiaddrs to connect to a private swarm instead of the public one, for example:
# INITIAL_PEERS = ['/ip4/10.1.2.3/tcp/31234/p2p/QmcXhze98AcgGQDDYna23s4Jho96n8wkwLJv78vxtFNq44']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from cpufeature import CPUFeature

    has_avx512 = CPUFeature["AVX512f"] and CPUFeature["OS_AVX512"]
except ImportError:
    has_avx512 = False

if DEVICE == "cuda":
    TORCH_DTYPE = "auto"
elif has_avx512:
    TORCH_DTYPE = torch.bfloat16
else:
    TORCH_DTYPE = torch.float32  # You can use bfloat16 in this case too, but it will be slow

STEP_TIMEOUT = 5 * 60
MAX_SESSIONS = 50  # Has effect only for API v1 (HTTP-based)
