# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py
import argparse
import json
import time
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

import torch
import uvicorn
from fastapi import Request, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse, Response
from packaging import version
from hivemind import get_logger
from transformers import GenerationConfig
import utils


from protocol import (
    CompletionRequest, CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    LogProbs, ModelCard, ModelList, ModelPermission, UsageInfo, random_uuid
)

try:
    import fastchat
    from fastchat.conversation import Conversation, SeparatorStyle
    from fastchat.model.model_adapter import get_conversation_template

    _fastchat_available = True
except ImportError:
    _fastchat_available = False

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = get_logger(__file__)

logger.info("Loading models")
models = utils.load_models()
app = FastAPI()


def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message,
                                      type="invalid_request_error").dict(),
                        status_code=status_code.value)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    if request.model in models:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )
    return ret


async def get_gen_prompt(request) -> str:
    if not _fastchat_available:
        raise ModuleNotFoundError(
            "fastchat is not installed. Please install fastchat to use "
            "the chat completion and conversation APIs: `$ pip install fschat`"
        )
    if version.parse(fastchat.__version__) < version.parse("0.2.23"):
        raise ImportError(
            f"fastchat version is low. Current version: {fastchat.__version__} "
            "Please upgrade fastchat to use: `$ pip install -U fschat`")

    conv = get_conversation_template(request.model)
    conv = Conversation(
        name=conv.name,
        system_template=conv.system_template,
        system_message="""A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n""",
        roles=conv.roles,
        messages=[],
        offset=conv.offset,
        sep_style=SeparatorStyle(conv.sep_style),
        sep=conv.sep,
        sep2=conv.sep2,
        stop_str=conv.stop_str,
        stop_token_ids=conv.stop_token_ids,
    )

    if isinstance(request.messages, str):
        prompt = request.messages
    else:
        for message in request.messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system_message = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    logger.info(f"Prompt: {prompt}")

    return prompt


async def check_length(
        request: Union[ChatCompletionRequest, CompletionRequest],
        model,
        tokenizer,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_ids: Optional[Union[List[int], List[List[int]]]] = None,
) -> Tuple[torch.LongTensor, Optional[JSONResponse]]:
    max_model_len = model.config.max_position_embeddings
    assert (not (prompt is None and prompt_ids is None)
            and not (prompt is not None and prompt_ids is not None)
            ), "Either prompt or prompt_ids should be provided."
    if prompt_ids is not None:
        input_ids = torch.LongTensor(prompt_ids)
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
    else:
        input_ids = torch.LongTensor(tokenizer(prompt).input_ids)
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)

    token_num = input_ids.shape[1]

    if request.max_tokens is None:
        request.max_tokens = max_model_len - token_num
    if token_num + request.max_tokens > max_model_len:
        return input_ids, create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return input_ids, None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = []
    for model_name in models:
        model_card = ModelCard(
            id=model_name,
            root=model_name,
            permission=[ModelPermission()],
        )
        model_cards.append(model_card)

    return ModelList(data=model_cards)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - presence_penalty (since the HF generation does not currently support it)
        - logit_bias
    """
    logger.info(f"Received chat completion request: {request}")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.logit_bias is not None and len(request.logit_bias) > 0:
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    model_name = request.model

    model, tokenizer, backend_config = models[model_name]

    prompt = await get_gen_prompt(request)
    token_ids, error_check_ret = await check_length(request, model, tokenizer, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())
    try:
        skip_special_tokens = request.skip_special_tokens
        spaces_between_special_tokens = request.spaces_between_special_tokens

        if request.presence_penalty > 0.0:
            logger.warning("Presence_penalty is not supported yet")

        generation_config = GenerationConfig(
            num_return_sequences=request.n,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
            do_sample=True if request.temperature is not None or request.top_p is not None or request.top_k is not None else False,
            num_beams=request.best_of if request.use_beam_search else 1,
            early_stopping=request.stop,
            repetition_penalty=1.0 + request.presence_penalty if request.presence_penalty else 1.0,
            # output_scores=True,  # Example: set to True if you want to output prediction scores
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    # Streaming response
    will_do_streaming = (request.stream
                         and (request.best_of is None or request.n == request.best_of)
                         and not request.use_beam_search)

    if will_do_streaming:
        def create_stream_response_json(
                index: int,
                text: str,
                finish_reason: Optional[str] = None,
        ) -> str:
            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(content=text),
                finish_reason=finish_reason,
            )
            response = ChatCompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=model_name,
                choices=[choice_data],
            )
            response_json = response.json(ensure_ascii=False)

            return response_json

        async def completion_stream_generator(model, token_ids) -> AsyncGenerator[str, None]:
            END_STRING = "data: [DONE]\n\n"
            n_input_tokens = token_ids.shape[1]
            max_new_tokens = generation_config.max_new_tokens
            total_generated_tokens = 0  # Initialize total generated token count

            with model.inference_session(max_length=max_new_tokens + n_input_tokens) as session:
                for i in range(request.n):
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(role="assistant"),
                        finish_reason=None,
                    )
                    chunk = ChatCompletionStreamResponse(id=request_id,
                                                         choices=[choice_data],
                                                         model=model_name)
                    data = chunk.json(exclude_unset=True, ensure_ascii=False)
                    yield f"data: {data}\n\n"

                while total_generated_tokens < max_new_tokens:
                    delta_q = []
                    delta = []

                    while len(delta_q) + len(delta) < max_new_tokens:
                        if total_generated_tokens >= max_new_tokens:
                            break  # Stop generating if the max token limit is reached

                        outputs = model.generate(
                            inputs=token_ids,
                            max_new_tokens=max_new_tokens - total_generated_tokens,
                            # Adjust max_new_tokens based on tokens already generated
                            session=session,
                        )
                        delta = outputs[0, n_input_tokens:].tolist()
                        total_generated_tokens += len(delta)

                        logger.info(f"Generated {len(delta)} tokens")
                        logger.info(f"Tokens generated: delta_q={repr(delta_q)}, delta={repr(delta)}")

                        outputs_text = utils.safe_decode(tokenizer, delta_q + delta)

                        if outputs_text[-10:].find("\ufffd") > -1:
                            # If there's a replacement character, keep getting more tokens
                            # until we can decode properly
                            delta_q += delta
                            logger.info(f"Retry generation {repr(outputs_text)}")
                        else:
                            response_json = create_stream_response_json(index=0, text=outputs_text)
                            logger.info(f"Generated {repr(outputs_text)}")
                            yield f"data: {response_json}\n\n"
                            delta_q = []  # Clear delta_q for the next iteration
                            token_ids = None  # Clear inputs for the next generation step

                        if len(delta) > 0 and delta[-1] == tokenizer.eos_token_id:
                            yield END_STRING
                            return

                yield END_STRING

        return StreamingResponse(completion_stream_generator(model, token_ids),
                                 media_type="text/event-stream")

    num_prompt_tokens = token_ids.shape[1]

    results_generator = model.generate(token_ids, **generation_config.to_diff_dict())

    choices = []
    for index, output_tokens in enumerate(results_generator):
        text = tokenizer.decode(
            output_tokens[num_prompt_tokens:],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

        choice_data = ChatCompletionResponseChoice(
            index=index,
            message=ChatMessage(role="assistant", content=text),
            finish_reason=None,
        )
        choices.append(choice_data)

    num_generated_tokens = sum(
        len(output) for output in results_generator)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")

    return response


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.

    NOTE: Currently we do not support the following features:
        - echo (since the Petals engine does not currently support
          getting the logprobs of prompt tokens)
        - suffix (the language models we currently support do not support
          suffix)
        - presence_penalty (since the HF generation does not currently support it)
        - logit_bias
    """
    logger.info(f"Received completion request: {request}")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.logit_bias is not None and len(request.logit_bias) > 0:
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"

    model, tokenizer, backend_config = models[model_name]

    use_token_ids = False
    if isinstance(request.prompt, list):
        if len(request.prompt) == 0:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "please provide at least one prompt")
        first_element = request.prompt[0]
        if isinstance(first_element, int):
            use_token_ids = True
            prompt = request.prompt
        elif isinstance(first_element, (str, list)):
            if len(request.prompt) > 1:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    "multiple prompts in a batch is not currently supported")
            use_token_ids = not isinstance(first_element, str)
            prompt = request.prompt[0]
    else:
        prompt = request.prompt

    if use_token_ids:
        token_ids, error_check_ret = await check_length(request, model, tokenizer, prompt_ids=prompt)
    else:
        token_ids, error_check_ret = await check_length(request, model, tokenizer, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    created_time = int(time.monotonic())
    try:
        skip_special_tokens = request.skip_special_tokens
        spaces_between_special_tokens = request.spaces_between_special_tokens

        if request.presence_penalty > 0.0:
            logger.warning("presence_penalty is not supported yet")

        generation_config = GenerationConfig(
            num_return_sequences=request.n,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
            do_sample=True if request.temperature is not None or request.top_p is not None or request.top_k is not None else False,
            num_beams=request.best_of if request.use_beam_search else 1,
            early_stopping=request.stop,
            repetition_penalty=1.0 + request.presence_penalty if request.presence_penalty else 1.0,
            # output_scores=True,  # Example: set to True if you want to output prediction scores
        )
    except ValueError as e:
        logger.error(f"Error in generation config: {e}")
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    # Similar to the OpenAI API, when n != best_of, we do not stream the
    # results. In addition, we do not stream the results when use beam search.
    will_do_streaming = (request.stream
                         and (request.best_of is None or request.n == request.best_of)
                         and not request.use_beam_search)

    if will_do_streaming:
        def create_stream_response_json(
                index: int,
                text: str,
                logprobs: Optional[LogProbs] = None,
                finish_reason: Optional[str] = None,
        ) -> str:
            choice_data = CompletionResponseStreamChoice(
                index=index,
                text=text,
                logprobs=logprobs,
                finish_reason=finish_reason,
            )
            response = CompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=model_name,
                choices=[choice_data],
            )
            response_json = response.json(ensure_ascii=False)

            return response_json

        async def completion_stream_generator(model, token_ids) -> AsyncGenerator[str, None]:
            END_STRING = "data: [DONE]\n\n"
            n_input_tokens = token_ids.shape[1]
            max_new_tokens = generation_config.max_new_tokens
            total_generated_tokens = 0  # Initialize total generated token count

            with model.inference_session(max_length=max_new_tokens + n_input_tokens) as session:
                while total_generated_tokens < max_new_tokens:
                    delta_q = []
                    delta = []

                    while len(delta_q) + len(delta) < max_new_tokens:
                        if total_generated_tokens >= max_new_tokens:
                            break  # Stop generating if the max token limit is reached

                        outputs = model.generate(
                            inputs=token_ids,
                            max_new_tokens=max_new_tokens - total_generated_tokens,
                            # Adjust max_new_tokens based on tokens already generated
                            session=session,
                        )
                        delta = outputs[0, n_input_tokens:].tolist()
                        total_generated_tokens += len(delta)

                        logger.info(f"Generated {len(delta)} tokens")
                        logger.info(f"Tokens generated: delta_q={repr(delta_q)}, delta={repr(delta)}")

                        outputs_text = utils.safe_decode(tokenizer, delta_q + delta)

                        if outputs_text[-10:].find("\ufffd") > -1:
                            # If there's a replacement character, keep getting more tokens
                            # until we can decode properly
                            delta_q += delta
                            logger.info(f"Retry generation {repr(outputs_text)}")
                        else:
                            response_json = create_stream_response_json(index=0, text=outputs_text)
                            logger.info(f"Generated {repr(outputs_text)}")
                            yield f"data: {response_json}\n\n"
                            delta_q = []  # Clear delta_q for the next iteration
                            token_ids = None  # Clear inputs for the next generation step

                        if len(delta) > 0 and delta[-1] == tokenizer.eos_token_id:
                            yield END_STRING
                            return

                # End of stream
                yield END_STRING

        return StreamingResponse(completion_stream_generator(model, token_ids), media_type="text/event-stream")

    logger.info(f"Generating with config {generation_config.to_diff_dict()}")

    results_generator = model.generate(token_ids, **generation_config.to_diff_dict())

    choices = []

    for index, output_tokens in enumerate(results_generator):
        text = tokenizer.decode(
            output_tokens,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

        logprobs = None
        choice_data = CompletionResponseChoice(
            index=index,
            text=text,
            logprobs=logprobs,
            finish_reason=None,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(token_ids)
    num_generated_tokens = sum(
        len(output) for output in results_generator)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = CompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")

    return response


# run flask as main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Petals OpenAI-Compatible REST full API server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")

    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE
    )
