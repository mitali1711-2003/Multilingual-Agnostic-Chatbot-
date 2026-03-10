from datetime import datetime

from flask_login import UserMixin
from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    Float,
)
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class User(UserMixin, Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def get_id(self):
        return str(self.id)


class FAQ(Base):
    """
    Stores multilingual FAQs loaded from Kaggle / admin CSV upload.
    Embeddings are stored as a JSON-encoded list of floats for retrieval.
    """

    __tablename__ = "faqs"

    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    language = Column(String(10), nullable=False)  # e.g. 'en', 'hi', 'mr'
    category = Column(String(64), nullable=True)
    campus = Column(String(64), nullable=True)  # Multi-campus: e.g. "Main Campus", "North Campus"
    embedding = Column(Text, nullable=True)  # JSON list of floats
    created_at = Column(DateTime, default=datetime.utcnow)


class Conversation(Base):
    """
    High-level conversation tied to a logged-in user.
    Used for multi-turn context and analytics.
    """

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    language = Column(String(10), nullable=True)
    campus = Column(String(64), nullable=True)  # Multi-campus support
    created_at = Column(DateTime, default=datetime.utcnow)


class Message(Base):
    """
    Individual message in a conversation.
    Stores sender, text, language, matched FAQ (if any) and feedback.
    """

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    sender = Column(String(10), nullable=False)  # "user" or "bot"
    text = Column(Text, nullable=False)
    language = Column(String(10), nullable=True)
    faq_id = Column(Integer, ForeignKey("faqs.id"), nullable=True)
    similarity = Column(Float, nullable=True)
    feedback = Column(String(16), nullable=True)  # "positive" or "negative"
    created_at = Column(DateTime, default=datetime.utcnow)

