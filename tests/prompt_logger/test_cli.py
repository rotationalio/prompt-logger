import sys
import json
import pytest
from pathlib import Path

from prompt_logger.logger import PromptLogger


@pytest.fixture
def test_db(tmp_path):
    return f"sqlite:///{tmp_path / 'test.db'}"
    # Cleanup is handled by the test_db fixture


def test_export_command(tmp_path, test_db, capsys):
    """Test the export command with various scenarios."""
    # Setup test data
    logger = PromptLogger(namespace="test-namespace", database=test_db)
    logger.save_interaction("test prompt 1", "test response 1")
    logger.save_interaction("test prompt 2", "test response 2")

    # Test successful export
    output_file = str(tmp_path / "test.jsonl")
    export_cmd = [
        "prompt_logger",
        "export",
        output_file,
        "--namespace",
        "test-namespace",
        "--database",
        test_db,
        "--type",
        "text",
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
        assert {record["prompt"] for record in records} == {
            "test prompt 1",
            "test prompt 2",
        }


def test_export_command_no_database(tmp_path):
    """Test that export command handles the case where the database does not exist."""
    output_file = str(tmp_path / "test.jsonl")
    export_cmd = [
        "prompt_logger",
        "export",
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
