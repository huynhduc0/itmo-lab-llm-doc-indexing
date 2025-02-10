from sqlalchemy import create_engine, Column, Integer, String, Text, PickleType
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Identity
import os

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "docs_assistant")
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432/{POSTGRES_DB}" 
print(DATABASE_URL)
engine = create_engine(DATABASE_URL)

Base = declarative_base()

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
    id = Column(Integer, primary_key=True)  # THIS LINE IS CRUCIAL
    doc_name = Column(String, unique=True) # Ensure doc_name is unique

class DocumentTOC(Base):
   __tablename__ = "document_toc"
   id = Column(Integer, primary_key=True)
   doc_name = Column(String, unique=True) # Link to doc, ensure doc_name is unique
   toc_items = Column(PickleType) # Store TOC as a list using PickleType

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)