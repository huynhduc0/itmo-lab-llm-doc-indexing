from sqlalchemy import create_engine, Column, Integer, String, PickleType, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
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

    messages = relationship("ChatMessage", back_populates="session") # Add relationship


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    session_key = Column(String, ForeignKey("chat_sessions.session_key")) #Change to ForeignKey
    role = Column(String)
    content = Column(String(4000))

    session = relationship("ChatSession", back_populates="messages")# Add relationship


class DocumentInfo(Base):
    __tablename__ = "document_info"
    id = Column(Integer, primary_key=True)
    doc_name = Column(String, unique=True)

    chunks = relationship("ChunkInfo", back_populates="document") # Add relationship


class DocumentTOC(Base):
    __tablename__ = "document_toc"
    id = Column(Integer, primary_key=True)
    doc_name = Column(String, ForeignKey("document_info.doc_name")) #Change to ForeignKey
    toc_items = Column(PickleType) #Will only use TOC Title

    document = relationship("DocumentInfo") # Add relationship

class ChunkInfo(Base):
    __tablename__ = "chunk_info"

    id = Column(Integer, primary_key=True)
    doc_name = Column(String, ForeignKey("document_info.doc_name"))  # Link to document
    toc_title = Column(String)  # TOC title for the chunk
    filename = Column(String)  # Filename of the chunked MD file

    document = relationship("DocumentInfo", back_populates="chunks") # relationship

Base.metadata.create_all(engine)  # Create tables if they don't exist