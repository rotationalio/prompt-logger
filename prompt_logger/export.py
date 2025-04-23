import json
from datetime import datetime
from dataclasses import dataclass

from prompt_logger.models.inference import remove_none_values


@dataclass
class PromptExport:
    """
    PromptExport is a denormalized representation of executed prompts in the database
    so they can be exported to a JSONL file.
    """

    id: str
    namespace: str
    model: str
    messages: list[dict]
    completions: list[dict]
    inference_on: datetime
    inference_seconds: float
    tools: list[dict] = None
    generation_kwargs: dict = None

    def to_json(self):
        """
        Convert the PromptExport object to a JSON string.
        """

        data = {
            "id": self.id,
            "namespace": self.namespace,
            "model": self.model,
            "messages": self.messages,
            "tools": self.tools,
            "generation_kwargs": self.generation_kwargs,
            "completions": self.completions,
            "inference_on": self.inference_on.isoformat(),
            "inference_seconds": self.inference_seconds,
        }
        return json.dumps(remove_none_values(data))
