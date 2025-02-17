from sqlalchemy import create_engine, Column, Integer, String, PickleType, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os

Base = declarative_base()

DB_TYPE = os.getenv("DB_TYPE", "sqlite")
DB_URI = os.getenv("DB_URI", "sqlite:///./doc_assistant.db")

engine = create_engine(DB_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True)
    session_key = Column(String, unique=True)
    selected_docs = Column(PickleType)
    messages = relationship("ChatMessage", back_populates="session")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    session_key = Column(String, ForeignKey("chat_sessions.session_key"))
    role = Column(String)
    content = Column(String(4000))
    session = relationship("ChatSession", back_populates="messages")

class DocumentInfo(Base):
    __tablename__ = "document_info"
    id = Column(Integer, primary_key=True)
    doc_name = Column(String, unique=True)
    chunks = relationship("ChunkInfo", back_populates="document")

class DocumentTOC(Base):
    __tablename__ = "document_toc"
    id = Column(Integer, primary_key=True)
    doc_name = Column(String, ForeignKey("document_info.doc_name"))
    toc_items = Column(PickleType)
    document = relationship("DocumentInfo")

class ChunkInfo(Base):
    __tablename__ = "chunk_info"
    id = Column(Integer, primary_key=True)
    doc_name = Column(String, ForeignKey("document_info.doc_name"))
    toc_title = Column(String)
    filename = Column(String)
    document = relationship("DocumentInfo", back_populates="chunks")

Base.metadata.create_all(engine)