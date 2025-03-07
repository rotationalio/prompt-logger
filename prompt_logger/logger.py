import os
import json
import uuid
from functools import wraps
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine.url import make_url

from prompt_logger.models import PromptRecord, Base


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
        namespace,
        database: str = "sqlite:///prompts.db",
        create_if_not_exists: bool = True,
    ):
        self.namespace = namespace
        self.database = database
        self._init_db(create_if_not_exists=create_if_not_exists)

    def _init_db(self, create_if_not_exists: bool = True):
        # TODO: Support remote databases
        url = make_url(self.database)
        print(url.database)
        if not os.path.exists(url.database) and not create_if_not_exists:
            raise ValueError(f"Database does not exist: {self.database}")

        self.engine = create_engine(self.database)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_interaction(self, prompt: str, response: str, namespace: str = None):
        Session = sessionmaker(bind=self.engine)
        session = Session()

        record = PromptRecord(
            id=str(uuid.uuid4()),
            namespace=namespace or self.namespace,
            prompt=prompt,
            response=response,
        )

        session.add(record)
        session.commit()
        session.close()

    def export_to_jsonl(self, output_file: str, namespace: str = None):
        """
        Export prompt records to a JSONL file.

        Parameters
        ----------
        output_file: str
            Path to the output JSONL file
        namespace: str, optional
            If provided, only export records from this namespace. If None, export all records.
        """
        Session = sessionmaker(bind=self.engine)
        session = Session()

        try:
            query = session.query(PromptRecord)
            if namespace:
                query = query.filter(PromptRecord.namespace == namespace)

            with open(output_file, "w") as f:
                for record in query:
                    # Convert datetime to ISO format string for JSON serialization
                    data = {
                        "id": record.id,
                        "namespace": record.namespace,
                        "prompt": record.prompt,
                        "response": record.response,
                        "timestamp": (
                            record.timestamp.isoformat() if record.timestamp else None
                        ),
                    }
                    f.write(json.dumps(data) + "\n")
        finally:
            session.close()


def capture(namespace: str, database: str = "sqlite:///prompts.db"):
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
    logger = PromptLogger(namespace, database=database)

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
