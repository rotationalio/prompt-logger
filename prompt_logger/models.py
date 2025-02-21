from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class PromptRecord(Base):
    __tablename__ = "prompts"

    id = Column(String, primary_key=True)
    namespace = Column(String)
    prompt = Column(String)
    response = Column(String)
    timestamp = Column(DateTime, default=datetime.now)
    # TODO: Save metadata like model, temperature, top_p, etc.
