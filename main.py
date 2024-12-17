from fastapi import FastAPI, Header, HTTPException, Depends, Request, Form, Path
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel
from enum import Enum
import uvicorn
import os
from dotenv import load_dotenv
import openai
from secrets import token_urlsafe
from fastapi.middleware.cors import CORSMiddleware

openai.api_key = os.getenv("OPENAI_API_KEY")

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{os.getenv('DB_USER', 'default_user')}:{os.getenv('DB_PASSWORD', 'default_password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'default_db')}"
)

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

#VALID_API_KEYS = os.getenv("API_KEYS", "your-api-key-1,your-api-key-2").split(",")

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

class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)  # A description/name for the key
    created_at = Column(TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    last_used = Column(TIMESTAMP, nullable=True)
    is_active = Column(Boolean, default=True)

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
    db = SessionLocal()
    try:
        db_key = db.query(APIKey).filter(
            APIKey.key == x_api_key,
            APIKey.is_active == True
        ).first()
        
        if not db_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid API Key"
            )
        
        # Update last used timestamp
        db_key.last_used = datetime.utcnow()
        db.commit()
        
        return x_api_key
    finally:
        db.close()

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
    db_text = TextEntry(
        english=None,
        spanish=None,
        portuguese=None,
        french=None,
        deutch=None,
        italian=None,
        apikey_requested=api_key
    )
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
    api_key: str = Depends(validate_api_key),
    filter_by_key: bool = True
):
    query = db.query(TextEntry)
    if filter_by_key:
        query = query.filter(TextEntry.apikey_requested == api_key)
    texts = query.all()
    return texts

@app.post("/translate-text/")
async def translate_text(
    text: str = Form(...),
    language: Language = Form(...),
    db: Session = Depends(get_db),
    api_key: str = Depends(validate_api_key)
):
    try:
        # Create new text entry
        db_text = TextEntry(
            apikey_requested=api_key
        )
        
        # Set original text
        setattr(db_text, language, text)
        
        # Get all available languages from the Language enum
        all_languages = [lang for lang in Language]
        translations = {}
        
        # Translate to all languages except the source language
        for target_lang in all_languages:
            if target_lang == language:
                continue  # Skip the source language
                
            prompt = f"Translate this text from {language} to {target_lang}:\n{text}"
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional translator."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            translated_text = response.choices[0].message.content
            
            # Save translation to database entry
            setattr(db_text, target_lang, translated_text)
            
            # Store translation in response dictionary
            translations[target_lang] = translated_text
        
        # Save to database
        db.add(db_text)
        db.commit()
        db.refresh(db_text)
        
        return {
            "original_text": text,
            "original_language": language,
            "translations": translations,
            "id": db_text.id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

@app.get("/admin/api-keys/")
async def list_api_keys(
    db: Session = Depends(get_db),
    admin_key: str = Header(..., alias="X-Admin-Key")  # Require admin key for management
):
    if admin_key != os.getenv("ADMIN_KEY", "your-admin-key"):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin key"
        )
    
    keys = db.query(APIKey).all()
    return keys

@app.post("/admin/api-keys/")
async def create_api_key(
    name: str = Form(...),
    db: Session = Depends(get_db),
    admin_key: str = Header(..., alias="X-Admin-Key")
):
    if admin_key != os.getenv("ADMIN_KEY", "your-admin-key"):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin key"
        )
    
    # Generate a new API key
    new_key = token_urlsafe(32)
    
    db_key = APIKey(
        key=new_key,
        name=name
    )
    
    db.add(db_key)
    db.commit()
    db.refresh(db_key)
    
    return {
        "id": db_key.id,
        "key": new_key,
        "name": name,
        "created_at": db_key.created_at
    }

@app.delete("/admin/api-keys/{key_id}")
async def delete_api_key(
    key_id: int,
    db: Session = Depends(get_db),
    admin_key: str = Header(..., alias="X-Admin-Key")
):
    if admin_key != os.getenv("ADMIN_KEY", "your-admin-key"):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin key"
        )
    
    db_key = db.query(APIKey).filter(APIKey.id == key_id).first()
    if not db_key:
        raise HTTPException(
            status_code=404,
            detail="API key not found"
        )
    
    db_key.is_active = False  # Soft delete
    db.commit()
    
    return {"message": "API key deactivated successfully"}

openai.api_key = os.getenv("OPENAI_API_KEY")

templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("traductor.html", {
        "request": request,
        # Add any additional context data here if needed
    })

@app.get("/admin", response_class=HTMLResponse)
async def admin_home(request: Request):
    return templates.TemplateResponse("admin_home.html", {"request": request})

@app.get("/admin/keys", response_class=HTMLResponse)
async def admin_keys(request: Request):
    return templates.TemplateResponse("admin_keys.html", {"request": request})

@app.get("/client", response_class=HTMLResponse)
async def client_dashboard(request: Request):
    return templates.TemplateResponse("client_dashboard.html", {"request": request})

@app.get("/api/admin/statistics")
async def get_statistics(
    db: Session = Depends(get_db),
    admin_key: str = Header(..., alias="X-Admin-Key")
):
    if admin_key != os.getenv("ADMIN_KEY"):
        raise HTTPException(status_code=401, detail="Invalid admin key")
    
    total_keys = db.query(APIKey).count()
    total_translations = db.query(TextEntry).count()
    active_today = db.query(APIKey).filter(
        APIKey.last_used >= datetime.utcnow() - timedelta(days=1)
    ).count()
    
    return {
        "total_keys": total_keys,
        "total_translations": total_translations,
        "active_today": active_today
    }

# Response models for better type hints and validation
class PendingTranslation(BaseModel):
    id: int
    original_language: str
    text: str
    missing_translations: List[str]
    created_at: Optional[datetime] = None  # Make it optional with default None
    api_key_name: Optional[str] = None     # Make it optional with default None

    class Config:
        from_attributes = True  # This is needed for SQLAlchemy model conversion

@app.get("/admin/pending-translations", response_model=List[PendingTranslation])
async def get_pending_translations(
    db: Session = Depends(get_db),
    admin_key: str = Header(..., alias="X-Admin-Key")
):
    if admin_key != os.getenv("ADMIN_KEY"):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin key"
        )
    
    try:
        # Get entries that might have pending translations
        entries = (
            db.query(TextEntry, APIKey.name.label('api_key_name'))
            .outerjoin(APIKey, TextEntry.apikey_requested == APIKey.key)  # Changed to outer join
            .all()
        )
        
        pending_translations = []
        languages = ["english", "spanish", "portuguese", "french", "deutch", "italian"]
        
        for entry, api_key_name in entries:
            # Find which language has content (original language)
            original_language = None
            original_text = None
            missing_translations = []
            
            # Check each language field
            for lang in languages:
                value = getattr(entry, lang)
                if value is not None and original_language is None:
                    original_language = lang
                    original_text = value
                elif value is None:
                    missing_translations.append(lang)
            
            # Only include entries that have missing translations
            if missing_translations and original_language and original_text:
                pending_translations.append(
                    {
                        "id": entry.id,
                        "original_language": original_language,
                        "text": original_text,
                        "missing_translations": missing_translations,
                        "created_at": entry.created_at,
                        "api_key_name": api_key_name or "Unknown"  # Provide default value
                    }
                )
        
        # Sort by creation date, newest first
        pending_translations.sort(key=lambda x: x.get('created_at') or datetime.min, reverse=True)
        
        return pending_translations
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching pending translations: {str(e)}"
        )

@app.post("/admin/translations/{entry_id}")
async def save_human_translation(
    entry_id: int,
    translation: dict,
    db: Session = Depends(get_db),
    admin_key: str = Header(..., alias="X-Admin-Key")
):
    if admin_key != os.getenv("ADMIN_KEY"):
        raise HTTPException(status_code=401, detail="Invalid admin key")
    
    entry = db.query(TextEntry).filter(TextEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    # Update the specified language field
    setattr(entry, translation["language"], translation["text"])
    entry.updated_at = datetime.utcnow()
    
    db.commit()
    
    return {"message": "Translation saved successfully"}

@app.get("/admin/human-translations", response_class=HTMLResponse)
async def human_translations_dashboard(request: Request):
    return templates.TemplateResponse("human_translations.html", {"request": request})

@app.get("/get-text/{text_id}")
async def get_text_by_id(
    text_id: int = Path(..., gt=0),  # Ensure positive integer
    db: Session = Depends(get_db),
    api_key: str = Header(..., alias="X-API-Key")
):
    # Validate API key
    db_api_key = db.query(APIKey).filter(APIKey.key == api_key).first()
    if not db_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    # Get the text entry
    text_entry = (
        db.query(TextEntry)
        .filter(TextEntry.id == text_id)
        .first()
    )

    # Check if text exists
    if not text_entry:
        raise HTTPException(
            status_code=404,
            detail="Text not found"
        )

    # Check if the API key has access to this text
    if text_entry.apikey_requested != api_key:
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to access this text"
        )

    # Convert to dictionary and add metadata
    result = {
        "id": text_entry.id,
        "translations": {
            "english": text_entry.english,
            "spanish": text_entry.spanish,
            "portuguese": text_entry.portuguese,
            "french": text_entry.french,
            "deutch": text_entry.deutch,
            "italian": text_entry.italian
        },
        "metadata": {
            "created_at": text_entry.created_at,
            "updated_at": text_entry.updated_at,
            "api_key_name": db_api_key.name
        }
    }

    # Remove None values from translations
    result["translations"] = {
        k: v for k, v in result["translations"].items() 
        if v is not None
    }

    return result

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)

