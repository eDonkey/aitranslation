from fastapi import FastAPI, Header, HTTPException, Depends, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from enum import Enum
import uvicorn
import os
from dotenv import load_dotenv
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

load_dotenv()

app = FastAPI()


# Database Configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{os.getenv('DB_USER', 'default_user')}:{os.getenv('DB_PASSWORD', 'default_password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'default_db')}"
)

# If using Heroku, modify the URL as Heroku provides it with 'postgres://' instead of 'postgresql://'
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define API Keys (in production, store these securely)
VALID_API_KEYS = os.getenv("API_KEYS", "your-api-key-1,your-api-key-2").split(",")

# Define valid languages
class Language(str, Enum):
    ENGLISH = "english"
    SPANISH = "spanish"
    PORTUGUESE = "portuguese"
    FRENCH = "french"
    DEUTCH = "deutch"
    ITALIAN = "italian"

# Database Model
class TextEntry(Base):
    __tablename__ = "text_entries"

    id = Column(Integer, primary_key=True, index=True)
    english = Column(Text, nullable=True)
    spanish = Column(Text, nullable=True)
    portuguese = Column(Text, nullable=True)
    french = Column(Text, nullable=True)
    deutch = Column(Text, nullable=True)
    italian = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at = Column(TIMESTAMP, nullable=True, onupdate=text('CURRENT_TIMESTAMP'))
    apikey_requested = Column(String, nullable=False)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API Key validation
def validate_api_key(x_api_key: str = Header(...)):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return x_api_key

# Request model
class TextRequest(BaseModel):
    text: str
    language: Language

@app.post("/save-text/")
async def save_text(
    request: TextRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(validate_api_key)
):
    # Create new text entry with all fields initially None
    db_text = TextEntry(
        english=None,
        spanish=None,
        portuguese=None,
        french=None,
        deutch=None,
        italian=None,
        apikey_requested=api_key
    )
    
    # Set the text for the specified language
    setattr(db_text, request.language.value, request.text)
    
    db.add(db_text)
    db.commit()
    db.refresh(db_text)
    
    return {
        "message": f"Text saved successfully in {request.language.value}",
        "id": db_text.id
    }

@app.get("/get-texts/")
async def get_texts(
    db: Session = Depends(get_db),
    api_key: str = Depends(validate_api_key)
):
    texts = db.query(TextEntry).all()
    return texts

@app.post("/translate-text/")
async def translate_text(
    text: str = Form(...),
    language: Language = Form(...),
    target_language: Language = Form(...),
    db: Session = Depends(get_db),
    api_key: str = Depends(validate_api_key)
):
    try:
        # Prepare the prompt for GPT
        prompt = f"Translate this text from {language} to {target_language}:\n{text}"
        
        # Call OpenAI API with new format
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt}
            ]
        )
        
        translated_text = response.choices[0].message.content

        # Save both original and translated text
        db_text = TextEntry(
            apikey_requested=api_key
        )
        
        # Set original text
        setattr(db_text, language, text)
        # Set translated text
        setattr(db_text, target_language, translated_text)
        
        db.add(db_text)
        db.commit()
        db.refresh(db_text)
        
        return {
            "original_text": text,
            "translated_text": translated_text,
            "from_language": language,
            "to_language": target_language,
            "id": db_text.id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)

