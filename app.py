import hivemind
import torch
from flask import Flask
from flask_sock import Sock
from transformers import BloomTokenizerFast

from petals import DistributedBloomForCausalLM

import config


logger = hivemind.get_logger(__file__)

logger.info(f"Loading tokenizer for {config.MODEL_NAME}")
tokenizer = BloomTokenizerFast.from_pretrained(config.MODEL_NAME)

logger.info(f"Loading model {config.MODEL_NAME}")
model = DistributedBloomForCausalLM.from_pretrained(config.MODEL_NAME, torch_dtype=config.TORCH_DTYPE)
model = model.to(config.DEVICE)

logger.info("Starting Flask app")
app = Flask(__name__)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}
sock = Sock(app)


@app.route("/")
def main_page():
    return app.send_static_file("index.html")


import http_api
import websocket_api
