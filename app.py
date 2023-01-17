import hivemind
import torch
from flask import Flask
from flask_sock import Sock
from transformers import BloomTokenizerFast

from petals import DistributedBloomForCausalLM

import config


logger = hivemind.get_logger(__file__)

models = {}
for model_name in config.MODEL_NAMES:
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)

    logger.info(f"Loading model {model_name}")
    model = DistributedBloomForCausalLM.from_pretrained(model_name, torch_dtype=config.TORCH_DTYPE)
    model = model.to(config.DEVICE)

    models[model_name] = model, tokenizer

logger.info("Starting Flask app")
app = Flask(__name__)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}
sock = Sock(app)


@app.route("/")
def main_page():
    return app.send_static_file("index.html")


import http_api
import websocket_api
