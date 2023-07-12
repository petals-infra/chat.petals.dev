import hivemind
from flask import Flask
from flask_cors import CORS
from flask_sock import Sock
from transformers import AutoTokenizer

from petals import AutoDistributedModelForCausalLM

import config


logger = hivemind.get_logger(__file__)

models = {}
for model_name in config.MODEL_NAMES:
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=False, use_fast=False)
    # We set use_fast=False since LlamaTokenizerFast takes a long time to init

    logger.info(f"Loading model {model_name} with dtype {config.TORCH_DTYPE}")
    if model_name == "artek0chumak/guanaco-65b":
        model = AutoDistributedModelForCausalLM.from_pretrained(
            "enoch/llama-65b-hf",
            active_adapter="artek0chumak/guanaco-65b",
            torch_dtype=config.TORCH_DTYPE,
            initial_peers=config.INITIAL_PEERS,
            max_retries=3,
        )
    else:
        model = AutoDistributedModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=config.TORCH_DTYPE,
            initial_peers=config.INITIAL_PEERS,
            max_retries=3,
        )
    model = model.to(config.DEVICE)

    models[model_name] = model, tokenizer

logger.info("Starting Flask app")
app = Flask(__name__)
CORS(app)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}
sock = Sock(app)


@app.route("/")
def main_page():
    return app.send_static_file("index.html")


import http_api
import websocket_api
