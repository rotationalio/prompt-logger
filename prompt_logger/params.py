from typing import Any, List, Dict
from dataclasses import dataclass


def get_param(kwargs: dict, param: str, required: bool = False, default=None):
    """
    Helper function to get a parameter from a dictionary of keyword arguments,
    handling defaults and errors.
    """

    if param not in kwargs and required:
        raise ValueError(f"Missing required keyword argument: {param}")
    return kwargs.pop(param, default)


@dataclass
class ChatCompletionParams:
    """
    Known parameters for the chat completion method which should be kept in sync with
    the OpenAI spec.
    """

    model: str
    messages: List
    tools: List[Dict] = None
    extra_kwargs: Dict[str, Any] = None

    @classmethod
    def from_kwargs(cls, **kwargs):
        """
        Create a ChatCompletionParams object from keyword arguments.
        """

        model = get_param(kwargs, "model", required=True)
        messages = get_param(kwargs, "messages", required=True)
        tools = get_param(kwargs, "tools")
        extra_kwargs = kwargs.copy()

        return cls(
            model=model,
            messages=messages,
            tools=tools,
            extra_kwargs=extra_kwargs,
        )
