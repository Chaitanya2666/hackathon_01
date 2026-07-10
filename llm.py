import json
import functools
import requests
from config import API_URL, API_KEY

session = requests.Session()


@functools.lru_cache(maxsize=256)
def _ask_llm_cached(messages_json: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
        "Ocp-Apim-Subscription-Key": API_KEY,
    }

    payload = {
        "messages": json.loads(messages_json),
        "max_completion_tokens": 4096,
    }

    response = session.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]

    print("\nStatus Code:", response.status_code)
    print("Finish Reason:", choice.get("finish_reason"))

    message = choice["message"].get("content", "")
    if not message or not message.strip():
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            return (
                "The model reached the maximum completion limit before "
                "producing an answer. Try reducing the input data or "
                "increasing max_completion_tokens."
            )
        return "The model returned an empty response."

    return message.strip()


def ask_llm(messages):
    try:
        messages_json = json.dumps(messages, sort_keys=True)
        return _ask_llm_cached(messages_json)
    except requests.exceptions.RequestException as e:
        return f"API Request Error: {e}"
    except Exception as e:
        return f"Unexpected Error: {e}"
