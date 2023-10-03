#!/usr/bin/env python
import json
import sys

# pip install websocket-client
import websocket

class ModelClient(object):
    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url
        self.ws = None
        self.model = None

    def open_session(self, model, max_length):
        self.ws = websocket.create_connection(self.endpoint_url)
        self.model = model
        payload = {
                "type": "open_inference_session",
                "model": self.model,
                "max_length": max_length,
            }
        self.ws.send(json.dumps(payload))
        assert json.loads(self.ws.recv())['ok'] == True

    def close_session(self):
        self.ws.close()

    def generate(self, prompt, **kwargs):
        payload = {
                "type": "generate",
                "inputs": prompt,
                "max_new_tokens": 1,
                "do_sample": 0,
                "temperature": 0,
                "stop_sequence": "</s>" if "bloomz" in self.model else "\n\n",
            }
        payload = {**payload, **kwargs}
        self.ws.send(json.dumps(payload))

        while True:
            data = json.loads(self.ws.recv())
            if not data['ok']:
                raise Exception(data['traceback'])
            yield data['outputs']
            if data['stop']:
                break
 
def main():
    client = ModelClient("ws://localhost:8000/api/v2/generate")
    # client = ModelClient("ws://chat.petals.ml/api/v2/generate")
    client.open_session("bigscience/bloom-petals", 128)

    if len(sys.argv) > 1:
        prompt = sys.argv[1]
        # Bloomz variant uses </s> instead of \n\n as an eos token
        if not prompt.endswith("\n\n"):
            prompt += "\n\n"
    else:
        prompt = "The SQL command to extract all the users whose name starts with A is: \n\n"
        print(f"Prompt: {prompt}")

    for out in client.generate(prompt,
                            do_sample=True,
                            temperature=0.75,
                            top_p=0.9):
        print(out, end="", flush=True)

    client.close_session()

if __name__ == '__main__':
    main()
