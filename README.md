# prompt-logger

A Python package for logging AI prompts to a database and exporting them for auditing, refinement, or analytics.

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

### As a Python Package

```python
from prompt_logger import PromptLogger, capture

# Initialize the logger
logger = PromptLogger("my-namespace")

# Log a single prompt and response
logger.save_interaction("What is the weather?", "It's sunny!")

# Export to JSONL
logger.export_to_jsonl("output.jsonl")

# Use the decorator to automatically log prompts
@capture("my-namespace")
def generate_text(prompt):
    # Your LLM call here
    return "Generated response"
```

### As a Command-Line Tool

Export recorded prompts to a JSONL file:

```bash
$ prompt-logger export output.jsonl --namespace my-namespace
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
