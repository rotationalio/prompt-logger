import json
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from prompt_logger.logger import PromptLogger, PromptRecord, capture, Base


@pytest.fixture
def test_db(tmp_path):
    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def logger(test_db):
    pl = PromptLogger("test-namespace", database=test_db)
    yield pl

    # Cleanup the database
    engine = create_engine(test_db)
    Base.metadata.drop_all(engine)


class TestPromptLogger:
    """
    Tests for the PromptLogger class.
    """

    def test_init(self, test_db):
        logger = PromptLogger("my-prompts", database=test_db)
        assert logger.namespace == "my-prompts"

    def test_logger_initialization(self, logger):
        assert logger.namespace == "test-namespace"
        assert isinstance(logger.engine, type(create_engine("sqlite:///")))

    def test_capture_decorator(self, test_db):
        @capture("namespace", database=test_db)
        def mock_llm(prompt):
            return "test response"

        result = mock_llm("test prompt")
        assert result == "test response"

        # Verify that the prompt was saved to the database
        session = sessionmaker(bind=create_engine(test_db))()
        record = session.query(PromptRecord).first()
        assert record.namespace == "namespace"
        assert record.prompt == "test prompt"
        assert record.response == "test response"
        session.close()

    def test_save_interaction(self, logger):
        logger.save_interaction("test prompt", "test response")

        session = sessionmaker(bind=logger.engine)()
        record = session.query(PromptRecord).first()
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
        logger.export_to_jsonl(str(output_file))

        # Verify the exported file
        with open(output_file) as f:
            lines = f.readlines()
            assert len(lines) == 2

            # Check first record
            record1 = json.loads(lines[0])
            assert record1["prompt"] == "prompt 1"
            assert record1["response"] == "response 1"
            assert record1["namespace"] == "test-namespace"
            assert "timestamp" in record1
            assert "id" in record1

            # Check second record
            record2 = json.loads(lines[1])
            assert record2["prompt"] == "prompt 2"
            assert record2["response"] == "response 2"
            assert record2["namespace"] == "test-namespace"
            assert "timestamp" in record2
            assert "id" in record2

    def test_export_to_jsonl_with_namespace(self, logger, tmp_path):
        # Add test data with different namespaces
        logger.save_interaction("prompt 1", "response 1", namespace="ns1")
        logger.save_interaction("prompt 2", "response 2", namespace="ns2")

        # Export only ns1 records
        output_file = tmp_path / "export.jsonl"
        logger.export_to_jsonl(str(output_file), namespace="ns1")

        # Verify only ns1 records were exported
        with open(output_file) as f:
            lines = f.readlines()
            assert len(lines) == 1

            record = json.loads(lines[0])
            assert record["prompt"] == "prompt 1"
            assert record["response"] == "response 1"
            assert record["namespace"] == "ns1"
