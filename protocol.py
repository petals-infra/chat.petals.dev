# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/protocol.py
import time
import uuid
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{random_uuid()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: str = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "petals"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = Field(default_factory=list)


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    model: str
    messages: Union[str, List[Dict[str, str]]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    max_tokens: int = 20
    stop: Optional[Union[str, bool]] = False
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Additional parameters supported by Petals
    best_of: Optional[int] = None
    top_k: Optional[int] = 50
    use_beam_search: Optional[bool] = False
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[List[int], List[List[int]], str, List[str]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    max_tokens: int = 20
    stop: Optional[Union[str, bool]] = False
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Additional parameters supported by Petals
    best_of: Optional[int] = None
    top_k: Optional[int] = 50
    use_beam_search: Optional[bool] = False
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True


class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str,
                                     float]]] = Field(default_factory=list)


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
