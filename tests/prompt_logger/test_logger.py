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
