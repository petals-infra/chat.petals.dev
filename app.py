import dataclasses
import json

import hivemind
from flask import Flask, render_template
from flask_cors import CORS
from flask_sock import Sock

import config
import utils

logger = hivemind.get_logger(__file__)

logger.info("Loading models")
models = utils.load_models()

logger.info("Starting Flask app")
app = Flask(__name__)
CORS(app)
app.config["SOCK_SERVER_OPTIONS"] = {"ping_interval": 25}
sock = Sock(app)


@app.route("/")
def main_page():
    default_family = list(config.MODEL_FAMILIES.values())[0]
    default_model = list(default_family)[0]

    model_configs = {
        model_config.backend.key: model_config
        for family_models in config.MODEL_FAMILIES.values()
        for model_config in family_models
    }

    return render_template(
        "index.html",
        default_model=default_model,
        model_families=config.MODEL_FAMILIES,
        model_config_json=json.dumps(model_configs, default=dataclasses.asdict),
    )


import http_api
import websocket_api
