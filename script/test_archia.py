import os
from openai import OpenAI

# Get token from environment
ARCHIA_TOKEN = os.environ.get("ARCHIA_TOKEN")

if not ARCHIA_TOKEN:
    raise ValueError("ARCHIA_TOKEN not found in environment variables.")

# Create client
client = OpenAI(
    base_url="https://registry.archia.app/v1",
    api_key="not-used",  # required but ignored
    default_headers={
        "Authorization": f"Bearer {ARCHIA_TOKEN}"
    },
)

# Make request
response = client.responses.create(
    model="priv-claude-sonnet-4-5-20250929",  # or any model you see from /v1/models
    input="Explain what Archia Cloud is in two sentences."
)

#print(ARCHIA_TOKEN)
# Print response
print(response.output[0].content[0].text)
