import hivemind
from flask import Flask
from flask_cors import CORS
from flask_sock import Sock

import utils
import views

logger = hivemind.get_logger(__file__)

logger.info("Loading models")
models = utils.load_models()

logger.info("Starting Flask app")
app = Flask(__name__)
CORS(app)
app.config["SOCK_SERVER_OPTIONS"] = {"ping_interval": 25}
sock = Sock(app)

logger.info("Pre-rendering index page")
index_html = views.render_index(app)


@app.route("/")
def main_page():
    return index_html


import http_api
import websocket_api
