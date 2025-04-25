import sys
import json
import pytest
from datetime import datetime
from unittest.mock import Mock
from pathlib import Path

from prompt_logger.logger import PromptLogger
from prompt_logger.models.inference import Model


@pytest.fixture
def test_db(tmp_path):
    return f"sqlite:///{tmp_path / 'test.db'}"
    # Cleanup is handled by the test_db fixture


def test_export_models(tmp_path, test_db, capsys):
    """Test the export command with various scenarios."""
    # Setup test data
    logger = PromptLogger(namespace="test-namespace", database=test_db)
    session = logger.Session()
    model = Model(
        name="test-model",
        version="1.0",
        provider="test-provider",
        description="test description",
        namespace="test-namespace",
    )
    model_2 = Model(
        name="test-model-2",
        version="1.0",
        provider="test-provider",
        description="test description",
        namespace="test-namespace",
    )
    session.add(model)
    session.add(model_2)
    session.commit()
    session.close()

    # Test successful export
    output_file = str(tmp_path / "test.jsonl")
    export_cmd = [
        "prompt_logger",
        "export",
        "models",
        output_file,
        "--namespace",
        "test-namespace",
        "--database",
        test_db,
    ]

    # Run the command
    from prompt_logger.cli import main

    sys.argv = export_cmd
    main()

    # Verify exported file contents
    with open(output_file) as f:
        lines = f.readlines()
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        assert all(record["namespace"] == "test-namespace" for record in records)
        assert all(
            record["name"] in ["test-model", "test-model-2"] for record in records
        )


def test_export_prompts(tmp_path, test_db, capsys):
    """Test the export command with various scenarios."""
    # Setup test data
    client = Mock()
    completion = Mock(
        choices=[
            Mock(
                message=Mock(
                    content="This is a test response.",
                    role="assistant",
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ]
    )
    client.models.retrieve = Mock(
        return_value=Mock(
            provider="test-provider",
            owned_by="test-org",
            created=datetime.now().timestamp(),
        ),
    )
    client.chat.completions.create.return_value = completion
    logger = PromptLogger(namespace="test-namespace", database=test_db)
    logger.attach_to_client(client)
    client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather in Minneapolis today?"},
        ],
    )

    # Test successful export
    output_file = str(tmp_path / "test.jsonl")
    export_cmd = [
        "prompt_logger",
        "export",
        "prompts",
        output_file,
        "--namespace",
        "test-namespace",
        "--database",
        test_db,
        "--type",
        "chat",
    ]

    # Run the command
    from prompt_logger.cli import main

    sys.argv = export_cmd
    main()

    # Verify exported file contents
    with open(output_file) as f:
        lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["namespace"] == "test-namespace"
        assert record["model"] == "gpt-4.1"
        assert record["messages"] == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather in Minneapolis today?"},
        ]
        assert record["completions"] == [
            {
                "role": "assistant",
                "content": "This is a test response.",
                "finish_reason": "stop",
            }
        ]
        assert record["inference_seconds"] is not None


def test_export_command_no_database(tmp_path):
    """Test that export command handles the case where the database does not exist."""
    output_file = str(tmp_path / "test.jsonl")
    export_cmd = [
        "prompt_logger",
        "export",
        "models",
        output_file,
        "--namespace",
        "test-namespace",
        "--database",
        "sqlite:///" + str(tmp_path / "missing.db"),
    ]

    # Run the command
    from prompt_logger.cli import main

    sys.argv = export_cmd
    with pytest.raises(ValueError):
        main()


def test_export_command_empty_database(tmp_path, test_db, capsys):
    """Test export command with an empty database."""
    _ = PromptLogger(namespace="test-namespace", database=test_db)

    output_file = str(tmp_path / "test.jsonl")
    export_cmd = [
        "prompt_logger",
        "export",
        "models",
        output_file,
        "--namespace",
        "test-namespace",
        "--database",
        test_db,
    ]

    # Run the command
    from prompt_logger.cli import main

    sys.argv = export_cmd
    main()

    # Verify empty file was created
    assert Path(output_file).exists()
    with open(output_file) as f:
        assert len(f.readlines()) == 0


def test_export_command_help(capsys):
    """Test that help message is displayed correctly."""
    help_cmd = [
        "prompt_logger",
        "export",
        "models",
        "--help",
    ]

    # Run the command
    from prompt_logger.cli import main

    sys.argv = help_cmd
    with pytest.raises(SystemExit):
        main()

    # Verify help message
    captured = capsys.readouterr()
    assert "usage: prompt_logger export" in captured.out
    assert "--namespace" in captured.out
    assert "--database" in captured.out
