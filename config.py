import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://apim-foundry-prod-ltts.azure-api.net/gpt5-mini/deployments/gpt-5-mini/chat/completions?api-version=2024-12-01-preview"

API_KEY = os.getenv("APIM_KEY")

print("=" * 50)
print("API URL :", API_URL)
print("API KEY :", repr(API_KEY))
print("=" * 50)