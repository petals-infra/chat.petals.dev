import requests


def test_api():
    request = {
        "model": "petals-team/StableBeluga2",
        "prompt": "A cat sat on a mat",
        "max_new_tokens": 20,
    }
    response = requests.post("http://localhost:8000/v1/completions", json=request)
    assert response.status_code == 200
    response.raise_for_status()
    assert "choices" in response.json(), "Response should contain a 'choices' field"
    assert isinstance(response.json()["choices"], list), "Choices should be a list"
    print("API test passed")
    print("Response:", response.json())
