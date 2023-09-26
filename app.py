import hivemind
from flask import Flask
from flask_cors import CORS
from flask_sock import Sock
from transformers import AutoTokenizer

from petals import AutoDistributedModelForCausalLM

import config


logger = hivemind.get_logger(__file__)

models = {}
for model_info in config.MODELS:
    logger.info(f"Loading tokenizer for {model_info.repo}")
    tokenizer = AutoTokenizer.from_pretrained(model_info.repo, add_bos_token=False, use_fast=False)

    logger.info(f"Loading model {model_info.repo} with adapter {model_info.adapter} and dtype {config.TORCH_DTYPE}")
    # We set use_fast=False since LlamaTokenizerFast takes a long time to init
    model = AutoDistributedModelForCausalLM.from_pretrained(
        model_info.repo,
        active_adapter=model_info.adapter,
        torch_dtype=config.TORCH_DTYPE,
        initial_peers=config.INITIAL_PEERS,
        max_retries=3,
    )
    model = model.to(config.DEVICE)

    model_name = model_info.name
    if model_name is None:  # Use default name based on model/repo repo
        model_name = model_info.adapter if model_info.adapter is not None else model_info.repo
    models[model_name] = model, tokenizer, model_info

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
