import threading
from contextlib import ExitStack, nullcontext
from traceback import format_exc
from uuid import uuid4

import hivemind
import torch
from flask import Flask, jsonify, request
from transformers import BloomTokenizerFast

from petals import DistributedBloomForCausalLM


MODEL_NAME = "bigscience/bloom-petals"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = None  # bfloat16 is default for bloom-petals
SESSION_EXPIRATION = 5 * 60
MAX_SESSIONS = 50

logger = hivemind.get_logger(__file__)


def load_model(model_name, device):
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)

    logger.info(f"Loading model {model_name}")
    model = DistributedBloomForCausalLM.from_pretrained(model_name, torch_dtype=TORCH_DTYPE)

    logger.info(f"Moving {model_name} to {device} device")
    model = model.to(device)

    return model, tokenizer


model, tokenizer = load_model(MODEL_NAME, DEVICE)

storage_lock = threading.Lock()
inference_sessions = hivemind.TimedStorage()  # Should be used under storage_lock

logger.info("Starting Flask app")
app = Flask(__name__)


@app.route("/")
def main_page():
    return app.send_static_file("index.html")


@app.get("/api/v1/open_inference_session")
def open_inference_session():
    try:
        max_length = get_typed_arg("max_length", int, 1024)

        with storage_lock:
            if len(inference_sessions) >= MAX_SESSIONS:
                raise RuntimeError(
                    f"Too many opened inference sessions (max {MAX_SESSIONS}), please come back later"
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
                hivemind.get_dht_time() + SESSION_EXPIRATION,
            )

        return jsonify(ok=True, session_id=session_id)
    except Exception:
        return jsonify(ok=False, traceback=format_exc())


@app.get("/api/v1/close_inference_session")
def close_inference_session():
    try:
        session_id = request.values.get("session_id")

        with storage_lock:
            del inference_sessions[session_id]

        return jsonify(ok=True, session_id=session_id)
    except Exception:
        return jsonify(ok=False, traceback=format_exc())


@app.post("/api/v1/generate")
def generate():
    try:
        inputs = request.values.get("inputs")
        do_sample = get_typed_arg("do_sample", int, 0)
        temperature = get_typed_arg("temperature", float, 1.0)
        top_k = get_typed_arg("top_k", int)
        top_p = get_typed_arg("top_p", float)
        max_length = get_typed_arg("max_length", int)
        max_new_tokens = get_typed_arg("max_new_tokens", int)
        session_id = request.values.get("session_id")
        logger.info(f"generate(), inputs={repr(inputs)}, session_id={session_id}")

        if inputs is not None:
            inputs = tokenizer(inputs, return_tensors="pt")["input_ids"].to(DEVICE)
            n_input_tokens = inputs.shape[1]
        else:
            n_input_tokens = 0

        if session_id is not None:
            with storage_lock:
                if session_id not in inference_sessions:
                    raise KeyError(f"Session {session_id} expired or does not exist")
                session, session_lock = inference_sessions.get(session_id).value
                inference_sessions.store(
                    session_id,
                    (session, session_lock),
                    hivemind.get_dht_time() + SESSION_EXPIRATION,
                )
            if session.position > 0:
                n_input_tokens += 1  # The .generate() will return the last token as well
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
        outputs = tokenizer.decode(outputs[0, n_input_tokens:])
        logger.info(f"generate(), outputs={repr(outputs)}")

        return jsonify(ok=True, outputs=outputs)
    except Exception:
        return jsonify(ok=False, traceback=format_exc())


def get_typed_arg(name, expected_type, default=None):
    value = request.values.get(name, default)
    return expected_type(value) if value is not None else None
