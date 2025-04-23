import json
import pytest
from datetime import datetime
from unittest.mock import Mock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dataclasses import dataclass, field

from prompt_logger.models.inference import (
    Prompt,
    TextCompletion,
    Model,
    TimestampedModel,
)
from prompt_logger.logger import PromptLogger, capture


@pytest.fixture
def test_db(tmp_path):
    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def logger(test_db):
    pl = PromptLogger(namespace="test-namespace", database=test_db)
    yield pl

    # Cleanup the database
    engine = create_engine(test_db)
    TimestampedModel.metadata.drop_all(engine)


@dataclass
class MockModel:
    owned_by: str = "test_org"
    created: int = datetime.now().timestamp()


@dataclass
class MockFunction:
    name: str = "weather_api"
    arguments: str = None


@dataclass
class MockToolCall:
    id: str = "tool_call_id"
    function: MockFunction = field(default_factory=MockFunction)


@dataclass
class MockMessage:
    role: str = "user"
    content: str = "this is a prompt"
    tool_calls: list = None


@dataclass
class MockCompletion:
    choices: list = None


@dataclass
class MockChoice:
    message: MockMessage = field(default_factory=MockMessage)
    finish_reason: str = "stop"


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def mock_client(mock_model):
    mock_client = Mock()
    mock_client.models.retrieve = Mock(return_value=mock_model)
    completion = MockCompletion(
        choices=[
            MockChoice(
                message=MockMessage(
                    role="assistant",
                    content="It's sunny!",
                ),
                finish_reason="stop",
            ),
        ]
    )
    mock_client.chat.completions.create = Mock(return_value=completion)
    return mock_client


class TestPromptLogger:
    """
    Tests for the PromptLogger class.
    """

    def test_init(self, test_db):
        logger = PromptLogger(namespace="my-prompts", database=test_db)
        assert logger.namespace == "my-prompts"

    def test_logger_initialization(self, logger):
        assert logger.namespace == "test-namespace"
        assert isinstance(logger.engine, type(create_engine("sqlite:///")))

    def test_attach_to_client(self, logger, mock_model):
        mock_client = Mock()
        mock_client.models.retrieve = Mock(return_value=mock_model)

        # Create mock responses for tool calling and assistant response
        assistant_completion = MockCompletion(
            choices=[
                MockChoice(
                    message=MockMessage(
                        role="assistant",
                        content="It's sunny!",
                    ),
                    finish_reason="stop",
                ),
                MockChoice(
                    message=MockMessage(
                        role="assistant",
                        content="It's rainy!",
                    ),
                    finish_reason="stop",
                ),
            ]
        )

        tool_completion = MockCompletion(
            choices=[
                MockChoice(
                    message=MockMessage(
                        role="tool",
                        content=None,
                        tool_calls=[
                            MockToolCall(
                                id="tool_call_id",
                                function=MockFunction(
                                    name="weather_api",
                                    arguments=json.dumps({"location": "Minneapolis"}),
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_call",
                ),
            ]
        )

        # Mock the completion responses
        mock_client.chat.completions.create = Mock(
            side_effect=[
                assistant_completion,
                tool_completion,
            ]
        )

        # Attach the logger to the mock client
        logger.attach_to_client(mock_client)

        # Invoke the model with a prompt
        prompts = [
            {
                "model": "gpt-3.5-turbo",
                "version": "3.5",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the weather?"},
                ],
                "temperature": 0.7,
            },
            {
                "model": "gemini-2.5",
                "version": "2.5",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the weather?"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather_api",
                            "description": "Get the current weather.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The location to get the weather for.",
                                    }
                                },
                                "required": ["location"],
                            },
                        },
                    }
                ],
                "tool_choice": "weather_api",
                "top_p": 0.9,
            },
        ]
        responses = [
            mock_client.chat.completions.create(**kwargs) for kwargs in prompts
        ]
        assert responses == [
            assistant_completion,
            tool_completion,
        ]

        # Check that the models were saved to the database
        session = sessionmaker(bind=logger.engine)()
        for prompt in prompts:
            model_name = prompt["model"]
            model = session.query(Model).filter_by(name=model_name).first()
            assert model is not None
            assert model.name == model_name
            assert model.version == prompt["version"]
            assert model.provider == mock_model.owned_by
            assert model.created is not None

        # Check that the assistant prompt was saved to the database
        prompt = (
            session.query(Prompt)
            .filter(Prompt.model.has(name=prompts[0]["model"]))
            .first()
        )
        assert prompt is not None
        assert prompt.namespace == "test-namespace"
        assert len(prompt.messages) == 2
        assert prompt.messages[0].role == prompts[0]["messages"][0]["role"]
        assert prompt.messages[0].content == prompts[0]["messages"][0]["content"]
        assert prompt.messages[1].role == prompts[0]["messages"][1]["role"]
        assert prompt.messages[1].content == prompts[0]["messages"][1]["content"]
        assert prompt.generation_kwargs == {
            "temperature": 0.7,
        }
        assert len(prompt.choices) == 2
        assert prompt.choices[0].role == assistant_completion.choices[0].message.role
        assert (
            prompt.choices[0].content == assistant_completion.choices[0].message.content
        )
        assert (
            prompt.choices[0].finish_reason
            == assistant_completion.choices[0].finish_reason
        )
        assert prompt.choices[1].role == assistant_completion.choices[1].message.role
        assert (
            prompt.choices[1].content == assistant_completion.choices[1].message.content
        )
        assert (
            prompt.choices[1].finish_reason
            == assistant_completion.choices[1].finish_reason
        )

        # Check that the tool prompt was saved to the database
        prompt = (
            session.query(Prompt)
            .filter(Prompt.model.has(name=prompts[1]["model"]))
            .first()
        )
        assert prompt is not None
        assert prompt.namespace == "test-namespace"
        assert len(prompt.messages) == 2
        assert prompt.messages[0].role == prompts[1]["messages"][0]["role"]
        assert prompt.messages[0].content == prompts[1]["messages"][0]["content"]
        assert prompt.messages[1].role == prompts[1]["messages"][1]["role"]
        assert prompt.messages[1].content == prompts[1]["messages"][1]["content"]
        assert prompt.generation_kwargs == {
            "tool_choice": "weather_api",
            "top_p": 0.9,
        }
        assert len(prompt.choices) == 1
        assert prompt.choices[0].role == tool_completion.choices[0].message.role
        assert prompt.choices[0].content == tool_completion.choices[0].message.content
        assert (
            prompt.choices[0].finish_reason == tool_completion.choices[0].finish_reason
        )
        assert len(prompt.tools) == 1
        assert prompt.tools[0].function_spec == prompts[1]["tools"][0]

        session.close()

    def test_capture_decorator(self, test_db):
        @capture("namespace", database=test_db)
        def mock_llm(prompt):
            return "test response"

        result = mock_llm("test prompt")
        assert result == "test response"

        # Verify that the prompt was saved to the database
        session = sessionmaker(bind=create_engine(test_db))()
        record = session.query(TextCompletion).first()
        assert record.namespace == "namespace"
        assert record.prompt == "test prompt"
        assert record.response == "test response"
        session.close()

    def test_save_interaction(self, logger):
        logger.save_interaction("test prompt", "test response")

        session = sessionmaker(bind=logger.engine)()
        record = session.query(TextCompletion).first()
        assert record is not None
        assert record.namespace == "test-namespace"
        assert record.prompt == "test prompt"
        assert record.response == "test response"
        session.close()

    def test_export_to_jsonl(self, logger, tmp_path):
        # Add some test data
        logger.save_interaction("prompt 1", "response 1")
        logger.save_interaction("prompt 2", "response 2")

        # Export to JSONL
        output_file = tmp_path / "export.jsonl"
        logger.export_text_prompts(str(output_file))

        # Verify the exported file
        with open(output_file) as f:
            lines = f.readlines()
            assert len(lines) == 2

            # Check first record
            record1 = json.loads(lines[0])
            assert record1["prompt"] == "prompt 1"
            assert record1["response"] == "response 1"
            assert record1["namespace"] == "test-namespace"
            assert "inference_on" in record1
            assert "id" in record1

            # Check second record
            record2 = json.loads(lines[1])
            assert record2["prompt"] == "prompt 2"
            assert record2["response"] == "response 2"
            assert record2["namespace"] == "test-namespace"
            assert "inference_on" in record2
            assert "id" in record2

    def test_export_to_jsonl_with_namespace(self, logger, tmp_path):
        # Add test data with different namespaces
        logger.save_interaction("prompt 1", "response 1", namespace="ns1")
        logger.save_interaction("prompt 2", "response 2", namespace="ns2")

        # Export only ns1 records
        output_file = tmp_path / "export.jsonl"
        logger.export_text_prompts(str(output_file), namespace="ns1")

        # Verify only ns1 records were exported
        with open(output_file) as f:
            lines = f.readlines()
            assert len(lines) == 1

            record = json.loads(lines[0])
            assert record["prompt"] == "prompt 1"
            assert record["response"] == "response 1"
            assert record["namespace"] == "ns1"

    def test_export_chat_prompts(self, logger, tmp_path, mock_client):
        # Add some test data
        logger.attach_to_client(mock_client)
        mock_client.chat.completions.create(
            messages=[{"role": "user", "content": "What is the weather today?"}],
            model="gpt-3.5-turbo",
            max_tokens=50,
        )

        # Export to JSONL
        output_file = tmp_path / "export.jsonl"
        logger.export_chat_prompts(str(output_file))

        # Verify the exported file
        with open(output_file) as f:
            lines = f.readlines()
            assert len(lines) == 1

            record = json.loads(lines[0])
            assert record["model"] == "gpt-3.5-turbo"
            assert record["messages"][0]["role"] == "user"
            assert record["messages"][0]["content"] == "What is the weather today?"
            assert record["completions"][0]["role"] == "assistant"
            assert record["completions"][0]["content"] == "It's sunny!"
            assert record["namespace"] == "test-namespace"

    def test_export_chat_prompts_tools(self, logger, tmp_path, mock_client):
        mock_client.chat.completions.create = Mock(
            side_effect=[
                MockCompletion(
                    choices=[
                        MockChoice(
                            message=MockMessage(
                                role="assistant",
                                tool_calls=[
                                    MockToolCall(
                                        id="tool_call_id",
                                        function=MockFunction(
                                            name="weather_api",
                                            arguments=json.dumps(
                                                {"location": "Minneapolis"}
                                            ),
                                        ),
                                    )
                                ],
                            ),
                            finish_reason="stop",
                        ),
                    ]
                ),
            ]
        )

        # Add some test data
        logger.attach_to_client(mock_client)
        mock_client.chat.completions.create(
            messages=[{"role": "user", "content": "What is the weather today?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "weather_api",
                        "description": "Get the current weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The location to get the weather for.",
                                }
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
            tool_choice="weather_api",
            model="gpt-3.5-turbo",
            max_tokens=50,
        )

        # Export to JSONL
        output_file = tmp_path / "export.jsonl"
        logger.export_chat_prompts(str(output_file))

        # Verify the exported file
        with open(output_file) as f:
            lines = f.readlines()
            assert len(lines) == 1

            record = json.loads(lines[0])
            assert record["model"] == "gpt-3.5-turbo"
            assert record["messages"][0]["role"] == "user"
            assert record["messages"][0]["content"] == "What is the weather today?"
            assert record["tools"][0]["function"]["name"] == "weather_api"
            assert record["generation_kwargs"] == {
                "tool_choice": "weather_api",
                "max_tokens": 50,
            }
            assert record["completions"][0]["role"] == "assistant"
            assert record["completions"][0]["tool_calls"][0]["id"] == "tool_call_id"
            assert (
                record["completions"][0]["tool_calls"][0]["function"]["name"]
                == "weather_api"
            )
            assert (
                record["completions"][0]["tool_calls"][0]["function"]["arguments"]
                == '{"location": "Minneapolis"}'
            )
            assert record["namespace"] == "test-namespace"
