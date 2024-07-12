import os
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from pandasai import llm

load_dotenv()


def call_openai(prompt: str) -> tuple:
    """Call the OpenAI API with a prompt.

    Args:
        prompt (str): The prompt to send to the API.

    Returns:
        chat_completion (openai.ChatCompletion): The completion object.
        content (str): The content of the completion.
    """

    # Create the client
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Send the message
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt,
                        "type": "text",
                    },
                ],
            },
        ],
        model=os.getenv("OPENAI_MODEL"),
    )

    # Get the content
    content = chat_completion.choices[0].message.content

    return chat_completion, content


def call_anthropic(prompt: str) -> tuple:
    """Call the Anthropic API with a prompt.

    Args:
        prompt (str): The prompt to send to the API.

    Returns:
        message (anthropic.Message): The message object.
        content (str): The content of the message.
    """

    # Create the client
    client = Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # Send the message
    message = client.messages.create(
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt,
                        "type": "text",
                    },
                ],
            },
        ],
        model=os.getenv("ANTHROPIC_MODEL"),
    )

    # Get the content
    content = message.content[0].text
    
    return message, content


def get_pandasai_openai_client() -> llm.OpenAI:
    """Get the PandasAI OpenAI client.

    Returns:
        client (llm.OpenAI): The PandasAI OpenAI client.
    """

    client = llm.OpenAI(
        api_token=os.getenv("OPENAI_API_KEY"),
    )

    return client


if __name__ == "__main__":
    
    prompt = "What is the meaning of life?"

    # OpenAI
    chat_completion, content = call_openai(prompt=prompt)
    print(content)

    # Anthropic
    message, content = call_anthropic(prompt=prompt)
    print(content)

    for row in content.split("\n\n"):
        print(row)

    # PandasAI
    client = get_pandasai_openai_client()

