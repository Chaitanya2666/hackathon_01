import requests
from config import API_URL, API_KEY


def ask_llm(messages):
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
        "Ocp-Apim-Subscription-Key": API_KEY
    }

    payload = {
        "messages": messages,
        "max_completion_tokens": 4096
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

        print("\nStatus Code:", response.status_code)

        if response.status_code != 200:
            print("Response:")
            print(response.text)
            response.raise_for_status()

        data = response.json()

        choice = data["choices"][0]

        # Print why generation stopped
        print("Finish Reason:", choice.get("finish_reason"))

        message = choice["message"].get("content", "")

        # Handle empty responses
        if not message or message.strip() == "":
            finish_reason = choice.get("finish_reason")

            if finish_reason == "length":
                return (
                    "The model reached the maximum completion limit before "
                    "producing an answer. Try reducing the input data or "
                    "increasing max_completion_tokens."
                )

            return "The model returned an empty response."

        return message.strip()

    except requests.exceptions.RequestException as e:
        return f"API Request Error: {e}"

    except Exception as e:
        return f"Unexpected Error: {e}"