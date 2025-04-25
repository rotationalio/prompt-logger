import json
from typing import List, Dict
from datetime import datetime
from dataclasses import dataclass

from prompt_logger.models.inference import remove_empty_values


@dataclass
class PromptExport:
    """
    PromptExport is a denormalized representation of executed prompts in the database
    so they can be exported to a JSONL file.
    """

    id: str
    namespace: str
    model: str
    messages: List[Dict]
    completions: List[Dict]
    inference_on: datetime
    inference_seconds: float
    tools: List[Dict] = None
    generation_kwargs: Dict = None

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
            "inference_on": self.inference_on.timestamp(),
            "inference_seconds": self.inference_seconds,
        }
        return json.dumps(remove_empty_values(data))
