from traceback import format_exc

import hivemind
from flask import jsonify, request

import config
from app import app, models

logger = hivemind.get_logger(__file__)


@app.post("/api/v1/generate")
def http_api_generate():
    try:
        model_name = get_typed_arg("model", str, config.DEFAULT_MODEL_NAME)
        inputs = request.values.get("inputs")
        do_sample = get_typed_arg("do_sample", int, 0)
        temperature = get_typed_arg("temperature", float, 1.0)
        top_k = get_typed_arg("top_k", int)
        top_p = get_typed_arg("top_p", float)
        max_length = get_typed_arg("max_length", int)
        max_new_tokens = get_typed_arg("max_new_tokens", int)
        session_id = request.values.get("session_id")
        logger.info(f"generate(), model={repr(model_name)}, inputs={repr(inputs)}")

        if session_id is not None:
            raise RuntimeError(
                "Reusing inference sessions was removed from HTTP API, please use WebSocket API instead"
            )

        model, tokenizer = models[model_name]
        fake_token = tokenizer("^")["input_ids"][0]  # Workaround to make SentencePiece .decode() keep leading spaces

        if inputs is not None:
            inputs = tokenizer(inputs, return_tensors="pt")["input_ids"].to(config.DEVICE)
            n_input_tokens = inputs.shape[1]
        else:
            n_input_tokens = 0

        outputs = model.generate(
            inputs=inputs,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        outputs = tokenizer.decode([fake_token] + outputs[0, n_input_tokens:].tolist())[1:]
        logger.info(f"generate(), outputs={repr(outputs)}")

        return jsonify(ok=True, outputs=outputs)
    except Exception:
        return jsonify(ok=False, traceback=format_exc())


def get_typed_arg(name, expected_type, default=None):
    value = request.values.get(name)
    return expected_type(value) if value is not None else default
