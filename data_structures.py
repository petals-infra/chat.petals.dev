from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class ModelBackendConfig:
    repository: str
    adapter: Optional[str] = None
    aliases: Sequence[str] = ()
    public_api: bool = True

    @property
    def key(self) -> str:
        return self.repository if self.adapter is None else self.adapter


@dataclass
class ModelFrontendConfig:
    name: str
    model_card: str
    license: str


@dataclass
class ModelChatConfig:
    max_session_length: int
    sep_token: str
    stop_token: str
    extra_stop_sequences: str
    generation_params: dict


@dataclass
class ModelConfig:
    backend: ModelBackendConfig
    frontend: ModelFrontendConfig
    chat: ModelChatConfig
