from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class ModelInfo:
    repository: str
    name: str
    model_card: str
    license: str

    adapter: Optional[str] = None
    aliases: Sequence[str] = ()
    public_api: bool = True

    @property
    def key(self) -> str:
        return self.repository if self.adapter is None else self.adapter
