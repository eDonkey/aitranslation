from fastapi import FastAPI, Header, HTTPException, Depends, Request, Form, Path, status, Query
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, text, Boolean, DateTime, func, Float, ForeignKey, desc, LargeBinary
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
from typing import Optional, List, Union
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
from pydantic import BaseModel
from enum import Enum
import uvicorn
import os
from dotenv import load_dotenv
import openai
from secrets import token_urlsafe
from fastapi.middleware.cors import CORSMiddleware
import logging
import io
import csv
from fastapi import File, UploadFile
from fastapi.responses import StreamingResponse
import asyncio

logger = logging.getLogger("uvicorn.error")

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

# Master API Key for administration
MASTER_API_KEY = os.getenv("MASTER_API_KEY")

# Define valid languages
class Language(str, Enum):
    ENGLISH = "english"
    SPANISH = "spanish"
    PORTUGUESE = "portuguese"
    FRENCH = "french"
    DEUTCH = "deutch"
    ITALIAN = "italian"
    FILIPINO = "filipino"
    JAPANESE = "japanese"
    VIETNAMESE = "vietnamese"

# Database Models
class TextEntry(Base):
    __tablename__ = "text_entries"

    id = Column(Integer, primary_key=True, index=True)
    apikey_requested = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)
    is_human_translation = Column(Boolean, default=False)

    # Relationship to translations
    translations = relationship("Translation", back_populates="text_entry")

class Translation(Base):
    __tablename__ = "translations"

    id = Column(Integer, primary_key=True, index=True)
    text_entry_id = Column(Integer, ForeignKey("text_entries.id"), nullable=False)
    language = Column(String, nullable=False)
    translated_text = Column(String, nullable=False)
    style = Column(String, nullable=True)
    model_version = Column(String, nullable=False)

    # Relationship to text entry
    text_entry = relationship("TextEntry", back_populates="translations")

class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    last_used = Column(TIMESTAMP, nullable=True)
    is_active = Column(Boolean, default=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    api_key = Column(String, unique=True, nullable=True)

class TranslationRequest(Base):
    __tablename__ = "translation_requests"

    id = Column(Integer, primary_key=True, index=True)
    source_language = Column(String, nullable=False)
    target_languages = Column(String, nullable=False)  # Comma-separated list
    translation_style = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    csv_file = Column(LargeBinary, nullable=False)  # Store the CSV as binary data
    cost = Column(Float, nullable=False, default=0.0)
    apikey_requested = Column(String, nullable=False)  # Added for user association

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
def validate_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
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

# Master API Key validation
def validate_master_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != MASTER_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid Master API Key"
        )
    return x_api_key

# Pydantic models
class TextRequest(BaseModel):
    text: str
    language: Language

class UserCreate(BaseModel):
    username: str
    email: str
    is_active: bool = True

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    is_active: Optional[bool] = None

class BillingPeriod(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"

class UserStatisticsRequest(BaseModel):
    user_id: Optional[int] = None
    period: BillingPeriod = BillingPeriod.MONTHLY
    start_date: Optional[str] = None  # Format: YYYY-MM-DD
    end_date: Optional[str] = None    # Format: YYYY-MM-DD

# Mount static files and templates (only for home page)
templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# HOME PAGE - Only interface remaining
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("traductor.html", {
        "request": request,
    })

# TRANSLATION ENDPOINTS

@app.post("/translate-text/")
async def translate_text(
    text: str = Form(...),
    source_language: Language = Form(...),
    target_languages: List[Language] = Form(None),
    translation_style: str = Form(None),
    db: Session = Depends(get_db),
    api_key: str = Depends(validate_api_key)
):
    """
    Translate text from source language to target languages
    """
    # Validate target languages
    all_languages = [lang for lang in Language]
    if target_languages:
        for lang in target_languages:
            if lang not in all_languages:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid target language: {lang}. Allowed languages are: {[l.value for l in all_languages]}"
                )
    else:
        # If no target languages are provided, translate to all languages except the source language
        target_languages = [lang for lang in all_languages if lang != source_language]
    
    # Translate the text
    translations = {}
    model_version = "gpt-4-turbo"
    try:
        for target_lang in target_languages:
            # Build the translation prompt
            style_prompt = f" in a {translation_style} style" if translation_style else ""
            prompt = f"Translate this text from {source_language.value} to {target_lang.value}{style_prompt}:\n{text}"
            
            response = openai.ChatCompletion.create(
                model=model_version,
                messages=[
                    {"role": "system", "content": "You are a professional translator."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            translated_text = response['choices'][0]['message']['content']
            translations[target_lang.value] = translated_text
        
        # Save the translations in the database
        db_text = TextEntry(
            apikey_requested=api_key,
            is_human_translation=False
        )
        db.add(db_text)
        db.commit()
        db.refresh(db_text)

        # Insert translations into the translations table
        for lang, translated_text in translations.items():
            db_translation = Translation(
                text_entry_id=db_text.id,
                language=lang,
                translated_text=translated_text,
                style=translation_style,
                model_version=model_version
            )
            db.add(db_translation)
        
        db.commit()
        
        return {
            "message": "Text translated successfully",
            "id": db_text.id,
            "original_text": text,
            "original_language": source_language.value,
            "translations": translations
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

@app.post("/translate-csv/")
async def translate_csv(
    file: bytes = File(...),
    source_language: Language = Form(...),
    target_languages: List[Language] = Form(None),
    translation_style: str = Form(None),
    db: Session = Depends(get_db),
    api_key: str = Depends(validate_api_key)
):
    """
    Translate CSV file from source language to target languages
    """
    # Validate target languages
    all_languages = [lang for lang in Language]
    if target_languages:
        for lang in target_languages:
            if lang not in all_languages:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid target language: {lang}. Allowed languages are: {[l.value for l in all_languages]}"
                )
    else:
        # If no target languages are provided, translate to all languages except the source language
        target_languages = [lang for lang in all_languages if lang != source_language]
    
    # Read the CSV file
    try:
        input_csv = io.StringIO(file.decode("utf-8"))
        reader = csv.reader(input_csv)
        rows = list(reader)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read the CSV file: {str(e)}"
        )
    
    if not rows or len(rows) < 1:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file is empty or invalid."
        )
    
    # The first column contains the texts to be translated
    texts = [row[0] for row in rows if row]
    translations = {lang.value: [] for lang in target_languages}
    model_version = "gpt-4-turbo"
    total_tokens_used = 0

    try:
        for target_lang in target_languages:
            # Build a single prompt for all texts in the batch
            style_prompt = f" in a {translation_style} style" if translation_style else ""
            prompt = f"Translate the following texts from {source_language.value} to {target_lang.value}{style_prompt}. Only return the translations, one per line, without any additional commentary:\n"
            prompt += "\n".join([f"- {text}" for text in texts])
            
            # Send a single API call for the batch
            response = openai.ChatCompletion.create(
                model=model_version,
                messages=[
                    {"role": "system", "content": "You are a professional translator."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Calculate tokens used
            total_tokens_used += response['usage']['total_tokens']
            
            # Parse the response into individual translations
            translated_texts = response['choices'][0]['message']['content'].split("\n")
            parsed_translations = [text.strip("- ").strip() for text in translated_texts if text.strip()]
            
            # Fill missing translations with a placeholder
            while len(parsed_translations) < len(texts):
                parsed_translations.append("Translation missing")
            translations[target_lang.value] = parsed_translations
        
        # Calculate the cost based on OpenAI's pricing
        cost_per_1k_tokens = 0.03
        total_cost = (total_tokens_used / 1000) * cost_per_1k_tokens
        
        # Write translations to the CSV
        output_csv = io.StringIO()
        writer = csv.writer(output_csv)
        
        # Write the header row
        header = ["Original"] + [lang.value for lang in target_languages]
        writer.writerow(header)
        
        # Write the original texts and their translations
        for i, text in enumerate(texts):
            row = [text] + [translations[lang.value][i] for lang in target_languages]
            writer.writerow(row)
        
        output_csv.seek(0)
        csv_data = output_csv.getvalue().encode("utf-8")
        
        # Save the request and CSV file in the database
        translation_request = TranslationRequest(
            source_language=source_language.value,
            target_languages=",".join([lang.value for lang in target_languages]),
            translation_style=translation_style,
            csv_file=csv_data,
            cost=total_cost,
            apikey_requested=api_key
        )
        db.add(translation_request)
        db.commit()
        
        # Return the updated CSV file
        return StreamingResponse(
            io.BytesIO(csv_data),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=translated.csv"}
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

@app.get("/api/translation/{translation_id}")
async def get_translation_by_id(
    translation_id: int,
    translation_type: str = Query("text", pattern="^(text|csv)$"),
    db: Session = Depends(get_db),
    api_key: str = Depends(validate_api_key)
):
    """
    Get a specific translation by ID
    """
    try:
        if translation_type == "csv":
            # Get CSV translation from TranslationRequest table
            translation_request = db.query(TranslationRequest)\
                .filter(
                    TranslationRequest.id == translation_id,
                    TranslationRequest.apikey_requested == api_key
                )\
                .first()
            
            if not translation_request:
                raise HTTPException(status_code=404, detail="CSV translation not found or access denied")
            
            return {
                "id": translation_request.id,
                "type": "csv",
                "source_language": translation_request.source_language,
                "target_languages": translation_request.target_languages.split(","),
                "translation_style": translation_request.translation_style,
                "cost": translation_request.cost,
                "created_at": translation_request.created_at,
                "csv_data": translation_request.csv_file.decode('utf-8') if translation_request.csv_file else None
            }
        
        else:  # translation_type == "text"
            # Get individual text translation
            text_entry = db.query(TextEntry)\
                .filter(
                    TextEntry.id == translation_id,
                    TextEntry.apikey_requested == api_key
                )\
                .first()
            
            if not text_entry:
                raise HTTPException(status_code=404, detail="Translation not found or access denied")
            
            # Get all translations for this text entry
            translations = db.query(Translation)\
                .filter(Translation.text_entry_id == text_entry.id)\
                .all()
            
            # Organize translations by language
            translation_data = {}
            for trans in translations:
                translation_data[trans.language] = {
                    "text": trans.translated_text,
                    "style": trans.style,
                    "model_version": trans.model_version
                }
            
            return {
                "id": text_entry.id,
                "type": "text",
                "translations": translation_data,
                "created_at": text_entry.created_at,
                "updated_at": text_entry.updated_at,
                "is_human_translation": text_entry.is_human_translation,
                "apikey_requested": text_entry.apikey_requested
            }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Get translation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/translation/{translation_id}/download")
async def download_translation_csv(
    translation_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(validate_api_key)
):
    """
    Download CSV file for bulk translations
    """
    try:
        # Get CSV translation from TranslationRequest table
        translation_request = db.query(TranslationRequest)\
            .filter(
                TranslationRequest.id == translation_id,
                TranslationRequest.apikey_requested == api_key
            )\
            .first()
        
        if not translation_request:
            raise HTTPException(status_code=404, detail="CSV translation not found or access denied")
        
        if not translation_request.csv_file:
            raise HTTPException(status_code=404, detail="CSV file not available")
        
        # Return the CSV file for download
        return StreamingResponse(
            io.BytesIO(translation_request.csv_file),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=translation_{translation_id}.csv"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Download CSV error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/translations/")
async def list_user_translations(
    translation_type: str = Query("all", pattern="^(all|text|csv)$"),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    api_key: str = Depends(validate_api_key)
):
    """
    List all translations for the authenticated user
    """
    try:
        result = {"text_translations": [], "csv_translations": []}
        
        if translation_type in ["all", "text"]:
            # Get text translations
            text_entries = db.query(TextEntry)\
                .filter(TextEntry.apikey_requested == api_key)\
                .order_by(TextEntry.created_at.desc())\
                .offset(offset)\
                .limit(limit)\
                .all()
            
            for entry in text_entries:
                # Get translations for this entry
                translations = db.query(Translation)\
                    .filter(Translation.text_entry_id == entry.id)\
                    .all()
                
                translation_data = {}
                for trans in translations:
                    translation_data[trans.language] = trans.translated_text
                
                result["text_translations"].append({
                    "id": entry.id,
                    "type": "text",
                    "translations": translation_data,
                    "created_at": entry.created_at,
                    "is_human_translation": entry.is_human_translation
                })
        
        if translation_type in ["all", "csv"]:
            # Get CSV translations
            csv_translations = db.query(TranslationRequest)\
                .filter(TranslationRequest.apikey_requested == api_key)\
                .order_by(TranslationRequest.created_at.desc())\
                .offset(offset)\
                .limit(limit)\
                .all()
            
            for csv_trans in csv_translations:
                result["csv_translations"].append({
                    "id": csv_trans.id,
                    "type": "csv",
                    "source_language": csv_trans.source_language,
                    "target_languages": csv_trans.target_languages.split(","),
                    "translation_style": csv_trans.translation_style,
                    "cost": csv_trans.cost,
                    "created_at": csv_trans.created_at
                })
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"List translations error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# USER MANAGEMENT ENDPOINTS (MASTER API KEY REQUIRED)

@app.post("/api/admin/users/")
async def create_user(
    user: UserCreate,
    db: Session = Depends(get_db),
    master_key: str = Depends(validate_master_key)
):
    """
    Create a new user (Master API key required)
    """
    # Check if username or email already exists
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Generate API key for the user
    api_key = token_urlsafe(32)
    
    # Create API key entry
    db_key = APIKey(
        key=api_key,
        name=f"Key for {user.username}",
        is_active=True,
        created_at=datetime.utcnow()
    )
    db.add(db_key)
    db.flush()
    
    # Create new user
    new_user = User(
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        api_key=api_key,
        created_at=datetime.utcnow()
    )
    
    # Link API key to user
    db_key.user_id = new_user.id
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {
        "id": new_user.id,
        "username": new_user.username,
        "email": new_user.email,
        "is_active": new_user.is_active,
        "api_key": api_key,
        "created_at": new_user.created_at
    }

@app.get("/api/admin/users/")
async def list_users(
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    master_key: str = Depends(validate_master_key)
):
    """
    List all users (Master API key required)
    """
    users = db.query(User).offset(offset).limit(limit).all()
    return [
        {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "is_active": user.is_active,
            "api_key": user.api_key,
            "created_at": user.created_at
        }
        for user in users
    ]

@app.get("/api/admin/users/{user_id}")
async def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    master_key: str = Depends(validate_master_key)
):
    """
    Get a specific user (Master API key required)
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_active": user.is_active,
        "api_key": user.api_key,
        "created_at": user.created_at
    }

@app.put("/api/admin/users/{user_id}")
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    master_key: str = Depends(validate_master_key)
):
    """
    Update a user (Master API key required)
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update only provided fields
    if user_update.username is not None:
        # Check if new username already exists
        existing_user = db.query(User).filter(
            User.username == user_update.username,
            User.id != user_id
        ).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        user.username = user_update.username
    
    if user_update.email is not None:
        # Check if new email already exists
        existing_user = db.query(User).filter(
            User.email == user_update.email,
            User.id != user_id
        ).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already exists")
        user.email = user_update.email
    
    if user_update.is_active is not None:
        user.is_active = user_update.is_active
        # Also update the API key status
        if user.api_key:
            api_key_obj = db.query(APIKey).filter(APIKey.key == user.api_key).first()
            if api_key_obj:
                api_key_obj.is_active = user_update.is_active
    
    db.commit()
    db.refresh(user)
    
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_active": user.is_active,
        "api_key": user.api_key,
        "created_at": user.created_at
    }

@app.delete("/api/admin/users/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    master_key: str = Depends(validate_master_key)
):
    """
    Delete a user (Master API key required)
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Deactivate the associated API key
    if user.api_key:
        api_key_obj = db.query(APIKey).filter(APIKey.key == user.api_key).first()
        if api_key_obj:
            api_key_obj.is_active = False
    
    db.delete(user)
    db.commit()
    
    return {"message": "User deleted successfully"}

@app.get("/api/admin/statistics")
async def get_statistics(
    db: Session = Depends(get_db),
    master_key: str = Depends(validate_master_key)
):
    """
    Get application statistics (Master API key required)
    """
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    total_api_keys = db.query(APIKey).count()
    active_api_keys = db.query(APIKey).filter(APIKey.is_active == True).count()
    total_text_translations = db.query(TextEntry).count()
    total_csv_translations = db.query(TranslationRequest).count()
    
    # Translations in the last 24 hours
    last_24_hours = datetime.utcnow() - timedelta(hours=24)
    recent_text_translations = db.query(TextEntry).filter(
        TextEntry.created_at >= last_24_hours
    ).count()
    recent_csv_translations = db.query(TranslationRequest).filter(
        TranslationRequest.created_at >= last_24_hours
    ).count()
    
    return {
        "users": {
            "total": total_users,
            "active": active_users
        },
        "api_keys": {
            "total": total_api_keys,
            "active": active_api_keys
        },
        "translations": {
            "text": {
                "total": total_text_translations,
                "last_24h": recent_text_translations
            },
            "csv": {
                "total": total_csv_translations,
                "last_24h": recent_csv_translations
            }
        }
    }

@app.get("/api/admin/users/{user_id}/statistics")
async def get_user_statistics(
    user_id: int,
    period: BillingPeriod = Query(BillingPeriod.MONTHLY),
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format. If provided, period will be calculated from this date."),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format. Only used when start_date is provided."),
    db: Session = Depends(get_db),
    master_key: str = Depends(validate_master_key)
):
    """
    Get detailed statistics for a specific user with billing analysis.
    
    Period calculation:
    - If start_date is provided:
      * If end_date is also provided: Uses exact date range
      * If only start_date: Calculates end_date based on period (monthly: +1 month, quarterly: +3 months, annually: +1 year)
    - If no start_date: Uses current date and goes backwards based on period
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Calculate date range based on period and provided dates
    if start_date:
        try:
            period_start = datetime.strptime(start_date, "%Y-%m-%d")
            
            if end_date:
                # Use exact date range if both dates provided
                period_end = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                # Calculate end date based on period from start date
                if period == BillingPeriod.MONTHLY:
                    period_end = period_start + relativedelta(months=1)
                elif period == BillingPeriod.QUARTERLY:
                    period_end = period_start + relativedelta(months=3)
                else:  # ANNUALLY
                    period_end = period_start + relativedelta(years=1)
                    
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail="Invalid date format. Use YYYY-MM-DD format."
            )
    else:
        # Default behavior: go backwards from current date
        period_end = datetime.utcnow()
        if period == BillingPeriod.MONTHLY:
            period_start = period_end - relativedelta(months=1)
        elif period == BillingPeriod.QUARTERLY:
            period_start = period_end - relativedelta(months=3)
        else:  # ANNUALLY
            period_start = period_end - relativedelta(years=1)
    
    # Ensure period_start is not after period_end
    if period_start > period_end:
        raise HTTPException(
            status_code=400,
            detail="Start date cannot be after end date"
        )
    
    # Get text translation statistics
    text_stats = db.query(
        func.count(TextEntry.id).label('total_translations'),
        func.count(func.distinct(func.date(TextEntry.created_at))).label('active_days')
    ).filter(
        TextEntry.apikey_requested == user.api_key,
        TextEntry.created_at >= period_start,
        TextEntry.created_at <= period_end
    ).first()
    
    # Get detailed text translation metrics
    text_translations = db.query(TextEntry).filter(
        TextEntry.apikey_requested == user.api_key,
        TextEntry.created_at >= period_start,
        TextEntry.created_at <= period_end
    ).all()
    
    # Count translations by language
    language_counts = {}
    total_characters_translated = 0
    
    for text_entry in text_translations:
        translations = db.query(Translation).filter(
            Translation.text_entry_id == text_entry.id
        ).all()
        
        for trans in translations:
            lang = trans.language
            if lang not in language_counts:
                language_counts[lang] = 0
            language_counts[lang] += 1
            total_characters_translated += len(trans.translated_text)
    
    # Get CSV translation statistics
    csv_stats = db.query(
        func.count(TranslationRequest.id).label('total_csv_translations'),
        func.sum(TranslationRequest.cost).label('total_cost'),
        func.avg(TranslationRequest.cost).label('avg_cost_per_csv')
    ).filter(
        TranslationRequest.apikey_requested == user.api_key,
        TranslationRequest.created_at >= period_start,
        TranslationRequest.created_at <= period_end
    ).first()
    
    # Get CSV details for analysis
    csv_translations = db.query(TranslationRequest).filter(
        TranslationRequest.apikey_requested == user.api_key,
        TranslationRequest.created_at >= period_start,
        TranslationRequest.created_at <= period_end
    ).all()
    
    # Calculate CSV metrics
    csv_rows_processed = 0
    csv_languages_used = set()
    
    for csv_trans in csv_translations:
        # Estimate rows from CSV data
        if csv_trans.csv_file:
            csv_content = csv_trans.csv_file.decode('utf-8')
            csv_rows_processed += len(csv_content.split('\n')) - 1  # Subtract header
        
        # Count unique target languages
        target_langs = csv_trans.target_languages.split(',')
        csv_languages_used.update(target_langs)
    
    # Calculate usage patterns
    daily_usage = {}
    for text_entry in text_translations:
        date_key = text_entry.created_at.strftime('%Y-%m-%d')
        if date_key not in daily_usage:
            daily_usage[date_key] = 0
        daily_usage[date_key] += 1
    
    for csv_trans in csv_translations:
        date_key = csv_trans.created_at.strftime('%Y-%m-%d')
        if date_key not in daily_usage:
            daily_usage[date_key] = 0
        daily_usage[date_key] += 1
    
    # Calculate peak usage
    peak_day_usage = max(daily_usage.values()) if daily_usage else 0
    avg_daily_usage = sum(daily_usage.values()) / len(daily_usage) if daily_usage else 0
    
    # Calculate the actual period length in days
    actual_period_days = (period_end - period_start).days
    
    # Prepare statistics data
    statistics = {
        "user_id": user_id,
        "username": user.username,
        "email": user.email,
        "period": period.value,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "actual_period_days": actual_period_days,
        "period_calculation_method": "custom_dates" if start_date else "backwards_from_now",
        "text_translations": {
            "total_translations": text_stats.total_translations or 0,
            "total_characters": total_characters_translated,
            "avg_characters_per_translation": total_characters_translated / (text_stats.total_translations or 1),
            "languages_used": language_counts,
            "unique_languages_count": len(language_counts),
            "active_days": text_stats.active_days or 0
        },
        "csv_translations": {
            "total_csv_files": csv_stats.total_csv_translations or 0,
            "total_rows_processed": csv_rows_processed,
            "total_cost": float(csv_stats.total_cost or 0),
            "avg_cost_per_csv": float(csv_stats.avg_cost_per_csv or 0),
            "languages_used": list(csv_languages_used),
            "unique_languages_count": len(csv_languages_used)
        },
        "usage_patterns": {
            "peak_day_usage": peak_day_usage,
            "avg_daily_usage": round(avg_daily_usage, 2),
            "total_active_days": len(daily_usage),
            "usage_consistency": round(avg_daily_usage / peak_day_usage, 2) if peak_day_usage > 0 else 0
        },
        "totals": {
            "total_api_calls": (text_stats.total_translations or 0) + (csv_stats.total_csv_translations or 0),
            "total_openai_cost": float(csv_stats.total_cost or 0),
            "estimated_text_cost": (text_stats.total_translations or 0) * 0.01  # Estimated cost per text translation
        }
    }
    
    # Generate AI-powered billing suggestion
    billing_suggestion = await generate_billing_suggestion(statistics, period)
    statistics["billing_analysis"] = billing_suggestion
    
    return statistics

@app.get("/api/admin/billing-analysis")
async def get_billing_analysis_all_users(
    period: BillingPeriod = Query(BillingPeriod.MONTHLY),
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format. If provided, period will be calculated from this date."),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format. Only used when start_date is provided."),
    db: Session = Depends(get_db),
    master_key: str = Depends(validate_master_key)
):
    """
    Get billing analysis for all users with flexible date range options
    """
    # Calculate date range (same logic as user statistics)
    if start_date:
        try:
            period_start = datetime.strptime(start_date, "%Y-%m-%d")
            
            if end_date:
                period_end = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                if period == BillingPeriod.MONTHLY:
                    period_end = period_start + relativedelta(months=1)
                elif period == BillingPeriod.QUARTERLY:
                    period_end = period_start + relativedelta(months=3)
                else:  # ANNUALLY
                    period_end = period_start + relativedelta(years=1)
                    
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail="Invalid date format. Use YYYY-MM-DD format."
            )
    else:
        period_end = datetime.utcnow()
        if period == BillingPeriod.MONTHLY:
            period_start = period_end - relativedelta(months=1)
        elif period == BillingPeriod.QUARTERLY:
            period_start = period_end - relativedelta(months=3)
        else:  # ANNUALLY
            period_start = period_end - relativedelta(years=1)
    
    if period_start > period_end:
        raise HTTPException(
            status_code=400,
            detail="Start date cannot be after end date"
        )
    
    users = db.query(User).filter(User.is_active == True).all()
    
    billing_summary = []
    total_revenue_potential = 0
    
    for user in users:
        try:
            # Get text translation statistics
            text_stats = db.query(
                func.count(TextEntry.id).label('total_translations')
            ).filter(
                TextEntry.apikey_requested == user.api_key,
                TextEntry.created_at >= period_start,
                TextEntry.created_at <= period_end
            ).first()
            
            # Get CSV translation statistics
            csv_stats = db.query(
                func.count(TranslationRequest.id).label('total_csv_translations'),
                func.sum(TranslationRequest.cost).label('total_cost')
            ).filter(
                TranslationRequest.apikey_requested == user.api_key,
                TranslationRequest.created_at >= period_start,
                TranslationRequest.created_at <= period_end
            ).first()
            
            total_api_calls = (text_stats.total_translations or 0) + (csv_stats.total_csv_translations or 0)
            total_openai_cost = float(csv_stats.total_cost or 0)
            
            # Simple billing calculation for summary
            if total_api_calls < 50:
                usage_tier = "light"
                suggested_charge = max(total_openai_cost * 1.5, 5.0)
            elif total_api_calls < 200:
                usage_tier = "moderate" 
                suggested_charge = max(total_openai_cost * 1.4, 15.0)
            elif total_api_calls < 1000:
                usage_tier = "heavy"
                suggested_charge = total_openai_cost * 1.3
            else:
                usage_tier = "enterprise"
                suggested_charge = total_openai_cost * 1.2
            
            billing_summary.append({
                "user_id": user.id,
                "username": user.username,
                "email": user.email,
                "suggested_charge": round(suggested_charge, 2),
                "usage_tier": usage_tier,
                "total_api_calls": total_api_calls,
                "openai_costs": total_openai_cost
            })
            
            total_revenue_potential += suggested_charge
                
        except Exception as e:
            print(f"Error processing user {user.id}: {str(e)}")
            continue
    
    # Sort by suggested charge (highest first)
    billing_summary.sort(key=lambda x: x["suggested_charge"], reverse=True)
    
    actual_period_days = (period_end - period_start).days
    
    return {
        "period": period.value,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "actual_period_days": actual_period_days,
        "period_calculation_method": "custom_dates" if start_date else "backwards_from_now",
        "total_users_analyzed": len(billing_summary),
        "total_revenue_potential": round(total_revenue_potential, 2),
        "avg_revenue_per_user": round(total_revenue_potential / len(billing_summary), 2) if billing_summary else 0,
        "user_billing_summary": billing_summary
    }

async def generate_billing_suggestion(statistics: dict, period: BillingPeriod) -> dict:
    """
    Use OpenAI to analyze user statistics and suggest billing amounts
    """
    try:
        # Prepare the data for AI analysis
        analysis_prompt = f"""
        Analyze the following user usage statistics and suggest a fair monthly billing amount:

        User Statistics for {period.value} period:
        - Text translations: {statistics['text_translations']['total_translations']}
        - Total characters translated: {statistics['text_translations']['total_characters']}
        - CSV files processed: {statistics['csv_translations']['total_csv_files']}
        - CSV rows processed: {statistics['csv_translations']['total_rows_processed']}
        - Total API calls: {statistics['totals']['total_api_calls']}
        - OpenAI costs incurred: ${statistics['totals']['total_openai_cost']}
        - Peak daily usage: {statistics['usage_patterns']['peak_day_usage']}
        - Average daily usage: {statistics['usage_patterns']['avg_daily_usage']}
        - Usage consistency: {statistics['usage_patterns']['usage_consistency']}
        - Active days: {statistics['usage_patterns']['total_active_days']}
        - Unique languages used: {statistics['text_translations']['unique_languages_count'] + statistics['csv_translations']['unique_languages_count']}

        Consider the following factors:
        1. Direct OpenAI API costs
        2. Server and infrastructure costs
        3. Profit margin (20-40%)
        4. User usage patterns and consistency
        5. Volume discounts for heavy users
        6. Competitive pricing in the translation industry

        Provide your analysis in the following JSON format (use actual numbers, not data types):
        {{
            "usage_tier": "light|moderate|heavy|enterprise",
            "cost_breakdown": {{
                "openai_costs": 0.00,
                "infrastructure_costs": 0.00,
                "profit_margin": 0.00,
                "total": 0.00
            }},
            "suggested_monthly_charge": 0.00,
            "reasoning": "detailed explanation of the pricing decision",
            "recommendations": ["list of recommendations for this user"]
        }}
        
        Important: Return only valid JSON with actual numeric values, not placeholder text.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a pricing analyst expert in SaaS billing and API services. Provide detailed, fair, and competitive pricing suggestions. Always return valid JSON with numeric values."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.3  # Lower temperature for more consistent pricing
        )
        
        # Parse the AI response
        ai_response = response['choices'][0]['message']['content']
        
        # Try to extract JSON from the response
        import json
        import re
        
        # Clean the response and try to parse JSON
        try:
            # Remove any markdown formatting
            cleaned_response = ai_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            billing_analysis = json.loads(cleaned_response)
            
            # Validate that we have the required fields
            if not all(key in billing_analysis for key in ["usage_tier", "cost_breakdown", "suggested_monthly_charge"]):
                raise ValueError("Missing required fields in AI response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing AI response: {e}")
            print(f"AI Response: {ai_response}")
            # Fallback to manual calculation
            billing_analysis = calculate_fallback_billing(statistics)
        
        # Adjust based on period
        if period == BillingPeriod.QUARTERLY:
            billing_analysis["suggested_charge"] = billing_analysis["suggested_monthly_charge"] * 3 * 0.95  # 5% discount for quarterly
        elif period == BillingPeriod.ANNUALLY:
            billing_analysis["suggested_charge"] = billing_analysis["suggested_monthly_charge"] * 12 * 0.85  # 15% discount for annual
        else:
            billing_analysis["suggested_charge"] = billing_analysis["suggested_monthly_charge"]
        
        return billing_analysis
        
    except Exception as e:
        print(f"Error generating AI billing suggestion: {str(e)}")
        # Fallback to manual calculation
        return calculate_fallback_billing(statistics)

def calculate_fallback_billing(statistics: dict) -> dict:
    """
    Fallback billing calculation if AI analysis fails
    """
    openai_costs = statistics['totals']['total_openai_cost']
    api_calls = statistics['totals']['total_api_calls']
    
    # Base infrastructure cost per API call
    infrastructure_cost_per_call = 0.002
    infrastructure_costs = api_calls * infrastructure_cost_per_call
    
    # Profit margin (30%)
    base_costs = openai_costs + infrastructure_costs
    profit_margin = base_costs * 0.3
    
    suggested_charge = base_costs + profit_margin
    
    # Determine usage tier
    if api_calls < 50:
        usage_tier = "light"
        suggested_charge = max(suggested_charge, 5.0)  # Minimum $5
    elif api_calls < 200:
        usage_tier = "moderate"
        suggested_charge = max(suggested_charge, 15.0)  # Minimum $15
    elif api_calls < 1000:
        usage_tier = "heavy"
        suggested_charge = suggested_charge * 0.95  # 5% volume discount
    else:
        usage_tier = "enterprise"
        suggested_charge = suggested_charge * 0.85  # 15% volume discount
    
    return {
        "usage_tier": usage_tier,
        "cost_breakdown": {
            "openai_costs": round(openai_costs, 2),
            "infrastructure_costs": round(infrastructure_costs, 2),
            "profit_margin": round(profit_margin, 2),
            "total": round(suggested_charge, 2)
        },
        "suggested_monthly_charge": round(suggested_charge, 2),
        "reasoning": f"Based on {api_calls} API calls with ${openai_costs:.2f} OpenAI costs, plus infrastructure and 30% margin",
        "recommendations": [
            "Monitor usage patterns for optimization opportunities",
            "Consider volume discounts for consistent high usage"
        ]
    }

@app.get("/api/admin/usage-trends")
async def get_usage_trends(
    days: int = Query(30, le=365),
    db: Session = Depends(get_db),
    master_key: str = Depends(validate_master_key)
):
    """
    Get usage trends across all users for the specified number of days
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Daily text translation counts
    daily_text_stats = db.query(
        func.date(TextEntry.created_at).label('date'),
        func.count(TextEntry.id).label('count'),
        func.count(func.distinct(TextEntry.apikey_requested)).label('unique_users')
    ).filter(
        TextEntry.created_at >= start_date,
        TextEntry.created_at <= end_date
    ).group_by(func.date(TextEntry.created_at)).all()
    
    # Daily CSV translation counts
    daily_csv_stats = db.query(
        func.date(TranslationRequest.created_at).label('date'),
        func.count(TranslationRequest.id).label('count'),
        func.sum(TranslationRequest.cost).label('total_cost'),
        func.count(func.distinct(TranslationRequest.apikey_requested)).label('unique_users')
    ).filter(
        TranslationRequest.created_at >= start_date,
        TranslationRequest.created_at <= end_date
    ).group_by(func.date(TranslationRequest.created_at)).all()
    
    # Combine and format the data
    trends = {}
    
    for stat in daily_text_stats:
        date_str = stat.date.strftime('%Y-%m-%d')
        trends[date_str] = {
            'date': date_str,
            'text_translations': stat.count,
            'csv_translations': 0,
            'total_cost': 0.0,
            'unique_users': stat.unique_users
        }
    
    for stat in daily_csv_stats:
        date_str = stat.date.strftime('%Y-%m-%d')
        if date_str in trends:
            trends[date_str]['csv_translations'] = stat.count
            trends[date_str]['total_cost'] = float(stat.total_cost or 0)
            trends[date_str]['unique_users'] = max(trends[date_str]['unique_users'], stat.unique_users)
        else:
            trends[date_str] = {
                'date': date_str,
                'text_translations': 0,
                'csv_translations': stat.count,
                'total_cost': float(stat.total_cost or 0),
                'unique_users': stat.unique_users
            }
    
    # Sort by date
    sorted_trends = sorted(trends.values(), key=lambda x: x['date'])
    
    return {
        "period_days": days,
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d'),
        "daily_trends": sorted_trends,
        "summary": {
            "total_text_translations": sum(t['text_translations'] for t in sorted_trends),
            "total_csv_translations": sum(t['csv_translations'] for t in sorted_trends),
            "total_openai_costs": sum(t['total_cost'] for t in sorted_trends),
            "peak_daily_usage": max((t['text_translations'] + t['csv_translations']) for t in sorted_trends) if sorted_trends else 0,
            "avg_daily_usage": sum((t['text_translations'] + t['csv_translations']) for t in sorted_trends) / len(sorted_trends) if sorted_trends else 0
        }
    }

# Add this import at the top with other imports
from fastapi import Request

# Add this new endpoint after your existing translation endpoints
@app.post("/demo-translate/")
async def demo_translate(
    request: Request,
    text: str = Form(...),
    source_language: Language = Form(...),
    target_language: Language = Form(...),
    db: Session = Depends(get_db)
):
    """
    Demo translation endpoint - No API key required
    Restrictions: 
    - Only 32 characters max
    - Only from server IP: 68.65.122.132 (where the page is hosted)
    - Single target language only
    """
    # Get the server's public IP address
    import socket
    import requests
    
    try:
        # Method 1: Try to get external IP using a service
        try:
            response = requests.get('https://api.ipify.org', timeout=5)
            server_ip = response.text.strip()
        except:
            # Method 2: Fallback - get local IP (might not be public IP)
            hostname = socket.gethostname()
            server_ip = socket.gethostbyname(hostname)
    except Exception:
        # Method 3: Final fallback - assume localhost
        server_ip = "127.0.0.1"
    
    allowed_ip = "54.227.3.30"
    
    # Check if the server is running from the allowed IP
    if server_ip != allowed_ip and server_ip != "127.0.0.1":  # Allow localhost for development
        raise HTTPException(
            status_code=403,
            detail=f"Demo access restricted. Server IP: {server_ip} is not authorized. This demo only works when hosted on {allowed_ip}."
        )
    
    # For development/testing, you might want to allow localhost
    # Remove this condition in production
    if server_ip == "127.0.0.1":
        print(f"Development mode: Server IP is {server_ip}, allowing demo access")
    
    # Check character limit
    if len(text) > 32:
        raise HTTPException(
            status_code=400,
            detail=f"Demo translation limited to 32 characters. Your text has {len(text)} characters."
        )
    
    # Validate that source and target languages are different
    if source_language == target_language:
        raise HTTPException(
            status_code=400,
            detail="Source and target languages must be different"
        )
    
    # Validate languages
    all_languages = [lang for lang in Language]
    if source_language not in all_languages or target_language not in all_languages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language. Allowed languages are: {[l.value for l in all_languages]}"
        )
    
    try:
        # Build the translation prompt
        prompt = f"Translate this text from {source_language.value} to {target_language.value}:\n{text}"
        
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a professional translator. Provide only the translation, no additional text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,  # Limit response for demo
            temperature=0.3
        )
        
        translated_text = response['choices'][0]['message']['content'].strip()
        
        # Log demo usage (optional - for analytics)
        demo_log = TextEntry(
            apikey_requested=f"DEMO_USER_FROM_{server_ip}",
            is_human_translation=False
        )
        db.add(demo_log)
        db.commit()
        db.refresh(demo_log)
        
        # Save the translation
        demo_translation = Translation(
            text_entry_id=demo_log.id,
            language=target_language.value,
            translated_text=translated_text,
            style="demo",
            model_version="gpt-4-turbo"
        )
        db.add(demo_translation)
        db.commit()
        
        return {
            "message": "Demo translation successful",
            "demo": True,
            "original_text": text,
            "original_language": source_language.value,
            "target_language": target_language.value,
            "translated_text": translated_text,
            "character_limit": "32 characters max",
            "note": "This is a demo. Sign up for full API access with unlimited translations.",
            "server_info": f"Demo served from IP: {server_ip}"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Demo translation failed: {str(e)}"
        )