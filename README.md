# Petals Chat

A chat [web app](http://chat.petals.ml) + a HTTP endpoint for BLOOM inference with the [Petals](https://petals.ml) client

## Interactive Chat

<div align="center">
<img src="https://i.imgur.com/p2nwiho.png" width="400px">
</div>

You can try it out [here](http://chat.petals.ml) or host the backend on your servers using these commands:

```bash
git clone https://github.com/borzunov/petals-chat.git
cd petals-chat
pip install -r requirements.txt
gunicorn app:app --bind 0.0.0.0:5000 --threads 10 --timeout 600
```

> **Note:** It is important to use `--threads` (not `--workers`), so reusing inference sessions works correctly.

## HTTP API Methods

If you develop your own web app, you can use our endpoint at `http://chat.petals.ml/api/v1/...` for research and development, then set up your own backend for production using the commands above. To do that, you need:

- A CPU-only server with 12+ GB RAM for the generation speed of 3-4 sec/token
- A GPU server with 8+ GB GPU VRAM for the generation speed of 1-1.5 sec/token

> **Note:** Please do not use our endpoint in production - it has a limited throughput, and we may pause or stop it any time.

### POST /api/v1/generate

Parameters:

- **inputs** (optional) - New user inputs. May be omitted if you continue generation in an inference session (see below).
- **do_sample** (optional) - If `0` (default), runs greedy generation. If `1`, performs sampling with parameters below.
- **temperature** (optional)
- **top_k** (optional)
- **top_p** (optional)
- **max_length** - Max length of generated text (including prefix) in tokens.
- **max_new_tokens** - Max number of newly generated tokens (excluding prefix).
- **session_id** (optional) - UUID of an inference session opened earlier (see methods below). This allows you to continue generation later without processing prefix from scratch.

Notes:

- You need to specify either `max_length` or `max_new_tokens`.
- If you'd like to solve downstream tasks in the zero-shot mode, start with `do_sample=0` (default).
- If you'd like to make a chat bot or write a long text, start with `do_sample=1, temperature=0.75, top_p=0.9` (tuning these params may help).

Returns (JSON):

- **ok** (bool)
- **outputs** (str)
- **traceback** (str) - the Python traceback if `ok == False`

### GET /api/v1/open_inference_session

Parameters:

- **max_length** (required)

Returns (JSON):

- **ok** (bool)
- **session_id** (str) - UUID of the opened session
- **traceback** (str) - the Python traceback if `ok == False`

### GET /api/v1/close_inference_session

Parameters:

- **session_id** (required)

Returns (JSON):

- **ok** (bool)
- **traceback** (str) - the Python traceback if `ok == False`
