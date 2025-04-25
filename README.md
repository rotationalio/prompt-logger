# prompt-logger

Prompt Logger reliably captures all the AI prompts you execute so you can focus on experimentation with very minimal instrumentation code.

## Features

- Programmatically log AI/chatbot interactions (prompts and responses) to a SQLite database
- Command-line interface to export logs to JSONL format
- Support for multiple user-defined namespaces
- Decorator for easy integration with existing code

## Installation

```bash
pip install prompt-logger
```

## Usage

### Working with OpenAI-style clients

If you're using the OpenAI client or a client that satisfies the `chat.completions.create` interface you can attach Prompt Logger to the client and it will automatically record all your prompts and their generated completions.

```python
from prompt_logger import PromptLogger
from openai import OpenAI

# Create the logger with a namespace and a database
logger = PromptLogger(namespace="my-namespace", database="sqlite:///my_prompts.db")

# Attach the logger to an AI client
client = OpenAI()
logger.attach_to_client(client)

# All completion requests to the client are logged
response = client.chat.completions.create(
   model="gpt-4.1",
   messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the weather today?"}
   ],
   max_tokens=50
)

# Export used models as a JSONL file
logger.export_models("models.jsonl")

# Export chat prompts to a JSONL file
logger.export_chat_prompts("prompts.jsonl")
```

Exported models contain some metadata about models used in completion requests.

```json
{"id": "74f0b720-dc78-491d-a123-2c33de50d2ee", "namespace": "my-namespace", "name": "gpt-4.1", "provider": "system", "created": 1744316542.0}
```

Exported prompts capture parameters used for completion requests and the generated completions.

```json
{"id": "d215d38f-ec59-4d20-9493-1c7f4e9a977f", "namespace": "default", "model": "gpt-4.1", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the weather today?"}], "generation_kwargs": {"max_tokens": 50}, "completions": [{"role": "assistant", "content": "I don't have access to real-time data or current weather updates. For today's weather, you can:\n\n- Check your preferred weather app (such as The Weather Channel, AccuWeather, or your phone's built-in weather app)\n- Search \"weather", "finish_reason": "length"}], "inference_on": 1745360143.533253, "inference_seconds": 1.272673}
```

### Working with custom code

You can use the `save_interaction` function to log one-off prompts.

```python
from prompt_logger import PromptLogger, capture

# Initialize the logger
logger = PromptLogger("my-namespace", database="sqlite:///my_prompts.db")

# Log a single prompt and response
logger.save_interaction("What is the weather?", "It's sunny!")

# Export to JSONL
logger.export_text_prompts("prompts.jsonl")
```

Or use the capture decorator to log prompts more automatically.

```python
# Use the decorator to automatically log prompts
@capture(namespace="my-namespace", database="sqlite:///my_prompts.db")
def generate_text(prompt):
    # Your LLM call here
    return "Generated response"
```

### Using the command line tool

You can use the command line tool to export models and prompts previously logged to the database.

```bash
$ prompt-logger export models models.jsonl --namespace=my-namespace --database=sqlite:///my_prompts.db
```

```bash
$ prompt-logger export prompts prompts.jsonl --namespace=my-namespace --database=sqlite:///my_prompts.db
```

## Development

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Run tests:
   ```bash
   pytest
   ```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
