import os
import json
import uuid
import logging
from functools import wraps
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine.url import make_url

from prompt_logger.params import ChatCompletionParams
from prompt_logger.models.inference import (
    TimestampedModel,
    Message,
    Prompt,
    Tool,
    TextCompletion,
    ToolCall,
    Model,
    ChatCompletion,
)
from prompt_logger.export import PromptExport


class PromptLogger:
    """
    PromptLogger is a wrapper around chatbot and LLM interfaces to capture and exfiltrate prompts and responses to a namespace.

    Usage
    -----
    Attach the logger to a method:
    ```
    logger = PromptLogger('my-prompts')
    client = ChatbotClient()
    logger.attach(client)
    ```

    One-off capture a prompt:
    ```
    logger = PromptLogger('my-prompts')
    client = ChatbotClient()
    response = logger.capture(client.text_generation, prompt='This is a test prompt')
    ```

    Parameters
    ----------
    namespace: str
        The namespace to log prompts and responses to.
    database: str (default: "sqlite:///prompts.db")
        The database connection string.
    create_if_not_exists: bool (default: True)
        If True, the database will be created if it does not exist.
    """

    def __init__(
        self,
        namespace: str = "default",
        database: str = "sqlite:///prompts.db",
        create_if_not_exists: bool = True,
        log_level: str = "INFO",
    ):
        self.namespace = namespace
        self.database = database
        self._init_db(create_if_not_exists=create_if_not_exists)
        self._init_logger(log_level=log_level)
        self._models = {}

    def _init_logger(self, log_level: str = "INFO"):
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

    def _init_db(self, create_if_not_exists: bool = True):
        # TODO: Support remote databases
        url = make_url(self.database)
        if not os.path.exists(url.database) and not create_if_not_exists:
            raise ValueError(f"Database does not exist: {self.database}")

        self.engine = create_engine(self.database)
        TimestampedModel.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def _fetch_model(self, client, model_name):
        """
        Fetch model info from the cache or the client.
        TODO: Add cache eviction policy.
        """
        if client in self._models:
            return self._models[client]
        try:
            model = client.models.retrieve(model_name)
        except Exception as e:
            self.logger.warning(
                f"Could not fetch model info, no metadata will be available: {e}"
            )
            return None
        self._models[client] = model
        return model

    def _get_target_fn(self, client, fn_name: str):
        """
        Helper to get a function from a client object by its fully qualified string.
        """

        fn = None
        for attr in fn_name.split("."):
            fn = getattr(fn or client, attr, None)
            if fn is None:
                raise ValueError(f"Could not find function to wrap: {fn_name}")

        if not callable(fn):
            raise ValueError(f"Function target is not callable: {fn_name}")
        return fn

    def attach_to_client(self, openai_client, wrapped_fn="chat.completions.create"):
        """
        Attach the logger to an OpenAI-style client to automatically capture prompts
        and responses.

        Parameters
        ----------
        openai_client: OpenAI-style client
            The client to attach the logger to which has a `chat.completions.create`
            method.

        wrapped_fn: str
            The function to wrap. Default is "chat.completions.create".

        Returns
        -------
        openai_client: OpenAI-style client
            The client with the logger attached.
        """

        target_fn = self._get_target_fn(openai_client, wrapped_fn)

        @wraps(target_fn)
        def wrapped_chat_completions_create(*args, **kwargs):
            try:
                params = ChatCompletionParams.from_kwargs(*args, **kwargs)
            except ValueError as e:
                raise ValueError(f"Could not parse parameters for {wrapped_fn}: {e}")

            # Attempt to fetch the model info from the client
            info = self._fetch_model(openai_client, params.model)

            # Check if the model already exists or we need to create it
            session = self.Session()
            model = (
                session.query(Model)
                .filter_by(
                    name=params.model,
                    version=params.extra_kwargs.get("version"),
                    namespace=self.namespace,
                )
                .first()
            )
            if model is None:
                model = Model(
                    name=params.model,
                    version=params.extra_kwargs.pop("version", None),
                    provider=info.owned_by if hasattr(info, "owned_by") else None,
                    description=params.extra_kwargs.pop("description", None),
                    created=(
                        datetime.fromtimestamp(info.created)
                        if hasattr(info, "created")
                        else None
                    ),
                    namespace=self.namespace,
                )

            # Create the prompt tools
            # TODO: Save tool functions in the database?
            tools = []
            if params.tools is not None:
                for tool in params.tools:
                    prompt_tool = (
                        session.query(Tool).filter_by(function_spec=tool).first()
                    )
                    if prompt_tool is None:
                        prompt_tool = Tool(
                            id=str(uuid.uuid4()),
                            function_spec=tool,
                            namespace=self.namespace,
                        )
                    tools.append(prompt_tool)

            # Add the model to the session
            session.add(model)

            # Create the prompt object
            prompt = Prompt(
                id=str(uuid.uuid4()),
                model=model,
                generation_kwargs=params.extra_kwargs,
                tools=tools,
                namespace=self.namespace,
                choices=[],
            )
            session.add(prompt)

            # Create the prompt messages
            messages = [
                Message(
                    id=str(uuid.uuid4()),
                    prompt=prompt,
                    role=msg["role"],
                    content=msg["content"],
                    tool_call_id=msg.get("tool_call_id", None),
                    namespace=self.namespace,
                )
                for msg in params.messages
            ]
            prompt.messages = messages

            # Call the original method
            start = datetime.now()
            completion = target_fn(*args, **kwargs)
            prompt.inference_seconds = (datetime.now() - start).total_seconds()

            # Create the response objects
            for choice in completion.choices:
                tool_calls = []
                if choice.message.tool_calls:
                    for tool_call in choice.message.tool_calls:
                        tc = ToolCall(
                            id=str(uuid.uuid4()),
                            tool_id=tool_call.id,
                            function_name=tool_call.function.name,
                            function_arguments=tool_call.function.arguments,
                            namespace=self.namespace,
                        )
                        tool_calls.append(tc)

                cc = ChatCompletion(
                    id=str(uuid.uuid4()),
                    prompt=prompt,
                    role=choice.message.role,
                    content=choice.message.content,
                    tool_calls=tool_calls,
                    finish_reason=choice.finish_reason,
                    namespace=self.namespace,
                )
                prompt.choices.append(cc)

            # Save models to the database
            session.commit()
            session.close()

            # Return the original response
            return completion

        # Replace the original method with the wrapped one
        openai_client.chat.completions.create = wrapped_chat_completions_create
        return openai_client

    def save_interaction(self, prompt: str, response: str, namespace: str = None):
        session = self.Session()
        completion = TextCompletion(
            namespace=namespace or self.namespace,
            prompt=prompt,
            response=response,
        )
        session.add(completion)
        session.commit()
        session.close()

    def export_chat_prompts(
        self,
        output_file: str,
        namespace: str = None,
        models: list = None,
    ):
        """
        Export executed prompts to a JSONL file.

        Parameters
        ----------
        output_file: str
            Path to the output JSONL file

        namespace: str, optional
            If provided, only export prompts from this namespace.

        models: list, optional
            If provided, only export prompts for these models.

        Returns
        -------
        int
            The number of prompts exported.
        """

        session = self.Session()
        try:
            query = session.query(Prompt).filter_by(
                namespace=namespace or self.namespace
            )
            if models:
                self.logger.info(f"Filtering prompts for models: {models}")
                query = query.filter(Prompt.model.has(Model.name.in_(models)))

            count = query.count()
            with open(output_file, "w") as f:
                for prompt in query:
                    export = PromptExport(
                        id=prompt.id,
                        namespace=prompt.namespace,
                        model=prompt.model.name,
                        messages=[msg.to_dict() for msg in prompt.messages],
                        tools=[
                            tool.function_spec
                            for tool in prompt.tools
                            if tool is not None
                        ],
                        generation_kwargs=prompt.generation_kwargs,
                        completions=[cc.to_dict() for cc in prompt.choices],
                        inference_on=prompt.created_at,
                        inference_seconds=prompt.inference_seconds,
                    )
                    f.write(export.to_json() + "\n")
        finally:
            session.close()
        self.logger.info(f"Exported {count} prompts to {output_file}")
        return count

    def export_text_prompts(
        self, output_file: str, namespace: str = None, models: list = None
    ):
        """
        Export text completion records to a JSONL file.

        Parameters
        ----------
        output_file: str
            Path to the output JSONL file
        namespace: str, optional
            If provided, only export records from this namespace. If None, export all records.
        models: list, optional
            If provided, only export records for these models.

        Returns
        -------
        int
            The number of records exported.
        """
        session = self.Session()

        try:
            query = session.query(TextCompletion).filter_by(
                namespace=namespace or self.namespace
            )
            if models:
                self.logger.info(f"Filtering prompts for models: {models}")
                query = query.filter(TextCompletion.model.in_(models))

            count = query.count()
            with open(output_file, "w") as f:
                for record in query:
                    f.write(json.dumps(record.to_dict()) + "\n")
        finally:
            session.close()
        self.logger.info(f"Exported {count} text completions to {output_file}")
        return count


def capture(namespace: str = "default", database: str = "sqlite:///prompts.db"):
    """
    Function decorator to capture interactions with a chatbot or LLM.

    Parameters
    ----------
    namespace: str
        The namespace to log prompts and responses to.
    database: str (default: "sqlite:///prompts.db")
        The database connection string.

    Usage
    -----
    ```
    @capture('my-prompts', database='sqlite:///my-prompts.db')
    def generate_text(prompt):
        ...
        return response
    ```
    """

    # TODO: Implement a default directory for the database (e.g. ~/.prompt_logger/prompts.db)
    logger = PromptLogger(namespace=namespace, database=database)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the prompt from the first arg or keyword arg
            prompt = args[0] if args else kwargs.get("prompt")

            # Call the original function
            response = func(*args, **kwargs)

            # Save the interaction to the database
            logger.save_interaction(prompt, response, namespace=namespace)
            return response

        return wrapper

    return decorator
