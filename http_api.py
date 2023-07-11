import threading
from contextlib import nullcontext
from traceback import format_exc
from uuid import uuid4

import hivemind
import torch
from flask import jsonify, request

import config
from app import app, models

logger = hivemind.get_logger(__file__)


storage_lock = threading.Lock()
inference_sessions = hivemind.TimedStorage()  # Should be used under storage_lock


@app.get("/api/v1/open_inference_session")
def http_api_open_inference_session():
    try:
        model_name = get_typed_arg("model", str, config.DEFAULT_MODEL_NAME)
        max_length = get_typed_arg("max_length", int, 1024)
        logger.info(f"open_inference_session(), model={repr(model_name)}, max_length={max_length}")

        model, _ = models[model_name]
        with storage_lock:
            if len(inference_sessions) >= config.MAX_SESSIONS:
                raise RuntimeError(
                    f"Too many opened inference sessions (max {config.MAX_SESSIONS}), please come back later"
                )
            # We don't release the lock here so that a concurrent thread else does not occupy our place.
            # session.__init__() and __enter__() are fast enough for that.

            session = model.inference_session(max_length=max_length)
            session.__enter__()
            session_lock = threading.Lock()

            session_id = uuid4().hex
            inference_sessions.store(
                session_id,
                (session, session_lock),
                hivemind.get_dht_time() + config.STEP_TIMEOUT,
            )

        return jsonify(ok=True, session_id=session_id)
    except Exception:
        return jsonify(ok=False, traceback=format_exc())


@app.get("/api/v1/close_inference_session")
def http_api_close_inference_session():
    try:
        session_id = request.values.get("session_id")
        logger.info(f"close_inference_session(), session_id={repr(session_id)}")

        with storage_lock:
            del inference_sessions[session_id]

        return jsonify(ok=True, session_id=session_id)
    except Exception:
        return jsonify(ok=False, traceback=format_exc())


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
        logger.info(f"generate(), model={repr(model_name)}, session_id={repr(session_id)}, inputs={repr(inputs)}")

        model, tokenizer = models[model_name]
        fake_token = tokenizer("^")["input_ids"][0]  # Workaround to make SentencePiece .decode() keep leading spaces

        if inputs is not None:
            inputs = tokenizer(inputs, return_tensors="pt")["input_ids"].to(config.DEVICE)
            n_input_tokens = inputs.shape[1]
        else:
            n_input_tokens = 0

        if session_id is not None:
            with storage_lock:
                if session_id not in inference_sessions:
                    raise KeyError(f"Session {repr(session_id)} expired or does not exist")
                session, session_lock = inference_sessions.get(session_id).value
                inference_sessions.store(
                    session_id,
                    (session, session_lock),
                    hivemind.get_dht_time() + config.STEP_TIMEOUT,
                )
        else:
            session = None
            session_lock = nullcontext()

        with session_lock:
            outputs = model.generate(
                inputs=inputs,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                session=session,
            )
        outputs = tokenizer.decode([fake_token] + outputs[0, n_input_tokens:].tolist())[1:]
        logger.info(f"generate(), outputs={repr(outputs)}")

        return jsonify(ok=True, outputs=outputs)
    except Exception:
        return jsonify(ok=False, traceback=format_exc())


def get_typed_arg(name, expected_type, default=None):
    value = request.values.get(name)
    return expected_type(value) if value is not None else default
