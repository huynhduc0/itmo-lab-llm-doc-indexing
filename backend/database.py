from sqlalchemy import create_engine, Column, Integer, String, PickleType
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()

# Database configuration
DB_TYPE = os.getenv("DB_TYPE", "sqlite")  # Default to SQLite if not specified
DB_URI = os.getenv("DB_URI", "sqlite:///./doc_assistant.db")  # SQLite default

# Create engine based on DB_TYPE and DB_URI
engine = create_engine(DB_URI)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True)
    session_key = Column(String, unique=True)
    selected_docs = Column(PickleType)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    session_key = Column(String)
    role = Column(String)
    content = Column(String)

class DocumentInfo(Base):
    __tablename__ = "document_info"
    id = Column(Integer, primary_key=True)
    doc_name = Column(String, unique=True)

class DocumentTOC(Base):
    __tablename__ = "document_toc"
    id = Column(Integer, primary_key=True)
    doc_name = Column(String, unique=True)
    toc_items = Column(PickleType)

Base.metadata.create_all(engine)  # Create tables if they don't exist