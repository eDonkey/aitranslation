from fastapi import FastAPI, Header, HTTPException, Depends, Request, Form, Path, status
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, text, Boolean, DateTime
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
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from starlette.status import HTTP_303_SEE_OTHER, HTTP_307_TEMPORARY_REDIRECT
import bcrypt

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
    english = Column(String)
    spanish = Column(String)
    portuguese = Column(String)
    french = Column(String)
    deutch = Column(String)
    italian = Column(String)
    apikey_requested = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)
    is_human_translation = Column(Boolean, default=False)

class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)  # A description/name for the key
    created_at = Column(TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    last_used = Column(TIMESTAMP, nullable=True)
    is_active = Column(Boolean, default=True)

class ModelPermission(str, Enum):
    AI_ONLY = "ai_only"
    HUMAN_ONLY = "human_only"
    BOTH = "both"
    NONE = "none"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    model_permission = Column(String, default=ModelPermission.NONE)
    created_at = Column(DateTime, default=datetime.utcnow)
    api_key = Column(String, unique=True, nullable=True)

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
    api_key: str = Header(..., alias="X-API-Key")
):
    try:
        # Check if this is a trial request
        if api_key == 'trial-key':
            # For trial requests, just do the translation without DB storage
            translations = {}
            all_languages = [lang for lang in Language]
            
            # Translate to all languages except the source language
            for target_lang in all_languages:
                if target_lang == language:
                    continue
                    
                prompt = f"Translate this text from {language} to {target_lang}:\n{text}"
                
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a professional translator."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                translated_text = response.choices[0].message.content
                translations[target_lang] = translated_text
            
            return {
                "original_text": text,
                "original_language": language,
                "translations": translations
            }
        
        # For authenticated requests, continue with the existing logic
        db_key = db.query(APIKey).filter(
            APIKey.key == api_key,
            APIKey.is_active == True
        ).first()
        
        if not db_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid API Key"
            )
        
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
                continue
                
            prompt = f"Translate this text from {language} to {target_lang}:\n{text}"
            
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

# Authentication constants and dependencies
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12
)

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    auto_error=False
)

# Authentication helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# User authentication functions
async def get_current_user(request: Request) -> Optional[User]:
    return getattr(request.state, "user", None)

async def get_current_active_admin(
    request: Request,
    current_user: Optional[User] = Depends(get_current_user)
) -> User:
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough privileges"
        )
    return current_user

# Authentication routes
@app.post("/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    access_token = create_access_token(data={"sub": user.username})
    
    # Create response based on user type
    response_data = {
        "access_token": access_token,
        "token_type": "bearer",
        "is_admin": user.is_admin,
        "username": user.username
    }
    
    response = JSONResponse(response_data)
    
    # Set cookie with token
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=1800  # 30 minutes
    )
    
    # If not admin, redirect to appropriate page
    if not user.is_admin:
        response.headers["Location"] = "/client"
        response.status_code = status.HTTP_303_SEE_OTHER
    
    return response

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    db: Session = Depends(get_db)
):
    # Get token from header or cookie
    token = None
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
    else:
        # Try to get token from cookie
        auth_cookie = request.cookies.get('access_token')
        if auth_cookie and auth_cookie.startswith('Bearer '):
            token = auth_cookie.split(' ')[1]

    if not token:
        return RedirectResponse(url="/login", status_code=303)

    try:
        # Decode token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            return RedirectResponse(url="/login", status_code=303)
        
        # Get user
        user = db.query(User).filter(User.username == username).first()
        if not user or not user.is_admin:
            return RedirectResponse(url="/login", status_code=303)
        
        # Get statistics
        pending_count = db.query(TextEntry).filter(
            TextEntry.updated_at == None
        ).count()
        
        active_keys = db.query(APIKey).filter(
            APIKey.is_active == True
        ).count()
        
        total_translations = db.query(TextEntry).count()
        
        # Get recent translations
        recent_translations = db.query(TextEntry)\
            .order_by(TextEntry.created_at.desc())\
            .limit(10)\
            .all()
        
        response = templates.TemplateResponse(
            "admin_dashboard.html",
            {
                "request": request,
                "pending_count": pending_count,
                "active_keys": active_keys,
                "total_translations": total_translations,
                "recent_translations": recent_translations,
                "user": user,  # Make sure to pass the user to the template
                "current_user": user  # Add this line to maintain consistency
            }
        )
        
        # Ensure token is refreshed in cookie
        response.set_cookie(
            key="access_token",
            value=f"Bearer {token}",
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=1800  # 30 minutes
        )
        
        return response
        
    except JWTError as e:
        print(f"JWT Error: {str(e)}")
        return RedirectResponse(url="/login", status_code=303)

    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        return RedirectResponse(url="/login", status_code=303)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN):
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers
            )
        return RedirectResponse(
            url="/login",
            status_code=status.HTTP_303_SEE_OTHER
        )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Bootstrap admin route for initial setup
@app.post("/bootstrap-admin")
async def bootstrap_admin(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    # Check if any users exist
    user_count = db.query(User).count()
    if user_count > 0:
        raise HTTPException(
            status_code=400,
            detail="Admin user already exists. Bootstrap is only for initial setup."
        )
    
    # Create the first admin user
    db_user = User(
        username=username,
        email=email,
        hashed_password=get_password_hash(password),
        is_admin=True,
        is_active=True
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return {"message": "Admin user created successfully"}

@app.get("/admin", response_class=HTMLResponse)
async def admin_home(
    request: Request,
    current_user: User = Depends(get_current_active_admin)
):
    return templates.TemplateResponse("admin_home.html", {
        "request": request,
        "user": current_user
    })

@app.get("/admin/keys", response_class=HTMLResponse)
async def admin_keys(
    request: Request,
    current_user: User = Depends(get_current_active_admin)
):
    return templates.TemplateResponse("admin_keys.html", {
        "request": request,
        "user": current_user
    })

@app.get("/client", response_class=HTMLResponse)
async def client_dashboard(
    request: Request,
    db: Session = Depends(get_db)
):
    # Get token from cookie
    auth_cookie = request.cookies.get('access_token')
    if not auth_cookie or not auth_cookie.startswith('Bearer '):
        return RedirectResponse(url="/login", status_code=303)
    
    token = auth_cookie.split(' ')[1]
    
    try:
        # Decode token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            return RedirectResponse(url="/login", status_code=303)
        
        # Get user
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return RedirectResponse(url="/login", status_code=303)

        # Get user's translations
        translations = db.query(TextEntry)\
            .filter(TextEntry.apikey_requested == user.api_key)\
            .order_by(TextEntry.created_at.desc())\
            .all()

        response = templates.TemplateResponse(
            "client_dashboard.html",  # Make sure this matches your template filename
            {
                "request": request,
                "user": user,
                "translations": translations or []  # Ensure translations is never None
            }
        )
        
        # Ensure token is refreshed in cookie
        response.set_cookie(
            key="access_token",
            value=f"Bearer {token}",
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=1800  # 30 minutes
        )
        
        return response
        
    except Exception as e:
        print(f"Client dashboard error: {str(e)}")  # Debug print
        return RedirectResponse(url="/login", status_code=303)

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
async def admin_human_translations(
    request: Request,
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    # Get pending human translations
    pending_translations = db.query(TextEntry)\
        .filter(TextEntry.is_human_translation == True)\
        .filter(TextEntry.updated_at == None)\
        .order_by(TextEntry.created_at.desc())\
        .all()
    
    # Get completed human translations
    completed_translations = db.query(TextEntry)\
        .filter(TextEntry.is_human_translation == True)\
        .filter(TextEntry.updated_at != None)\
        .order_by(TextEntry.updated_at.desc())\
        .all()
    
    return templates.TemplateResponse(
        "admin_human_translations.html",
        {
            "request": request,
            "user": current_user,
            "pending_translations": pending_translations,
            "completed_translations": completed_translations
        }
    )

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

@app.post("/admin/create-user")
async def create_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    is_admin: bool = Form(False),
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    db_user = User(
        username=username,
        email=email,
        hashed_password=get_password_hash(password),
        is_admin=is_admin
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"message": "User created successfully"}

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users_page(
    request: Request,
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    users = db.query(User).all()
    return templates.TemplateResponse(
        "admin_users.html",
        {
            "request": request,
            "users": users,
            "current_user": current_user
        }
    )

@app.post("/api/admin/users")
async def create_new_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    is_admin: bool = Form(False),
    model_permission: ModelPermission = Form(ModelPermission.BOTH),
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    try:
        # Check if username or email already exists
        if db.query(User).filter(User.username == username).first():
            raise HTTPException(status_code=400, detail="Username already registered")
        if db.query(User).filter(User.email == email).first():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Generate API key if permissions are granted
        api_key = None
        if model_permission != ModelPermission.NONE:
            api_key = token_urlsafe(32)
            
            # Create API key entry
            db_key = APIKey(
                key=api_key,
                name=f"Key for {username}",
                is_active=True,
                created_at=datetime.utcnow()
            )
            db.add(db_key)
            try:
                db.flush()  # Ensure the API key is created before linking to user
            except Exception as e:
                db.rollback()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error creating API key: {str(e)}"
                )
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            hashed_password=get_password_hash(password),
            is_admin=is_admin,
            is_active=True,
            model_permission=model_permission,
            api_key=api_key,  # Link the API key to the user
            created_at=datetime.utcnow()
        )
        
        db.add(new_user)
        try:
            db.commit()
            db.refresh(new_user)
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Error creating user: {str(e)}"
            )
        
        return {
            "message": "User created successfully",
            "api_key": api_key,
            "permissions": model_permission
        }
        
    except Exception as e:
        print(f"Error creating user: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error creating user: {str(e)}"
        )

@app.put("/api/admin/users/{user_id}")
async def update_user(
    user_id: int,
    is_active: bool = Form(...),
    is_admin: bool = Form(...),
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    # Prevent self-modification
    if user_id == current_user.id:
        raise HTTPException(
            status_code=400,
            detail="Cannot modify your own admin status"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.is_active = is_active
    user.is_admin = is_admin
    db.commit()
    return {"message": "User updated successfully"}

@app.delete("/api/admin/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    # Prevent self-deletion
    if user_id == current_user.id:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete your own account"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}

@app.middleware("http")
async def authenticate_user_middleware(request: Request, call_next):
    # Skip authentication for login and static routes
    if request.url.path in ["/login", "/token"] or request.url.path.startswith("/static"):
        return await call_next(request)

    # Get token from header or cookie
    token = None
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
    else:
        auth_cookie = request.cookies.get('access_token')
        if auth_cookie and auth_cookie.startswith('Bearer '):
            token = auth_cookie.split(' ')[1]

    if not token:
        if request.url.path.startswith("/admin"):
            return RedirectResponse(url="/login", status_code=303)
        return await call_next(request)

    try:
        # Verify token and get user
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username:
            db = SessionLocal()
            try:
                user = db.query(User).filter(User.username == username).first()
                if user and user.is_active:
                    # Attach user to request state
                    request.state.user = user
                    response = await call_next(request)
                    
                    # Refresh token in cookie
                    response.set_cookie(
                        key="access_token",
                        value=f"Bearer {token}",
                        httponly=True,
                        secure=True,
                        samesite="lax",
                        max_age=1800  # 30 minutes
                    )
                    return response
            finally:
                db.close()

    except JWTError:
        pass

    if request.url.path.startswith("/admin"):
        return RedirectResponse(url="/login", status_code=303)
    return await call_next(request)

# @app.on_event("startup")
# async def alter_table():
#     engine = create_engine(DATABASE_URL)
#     with engine.connect() as connection:
#         try:
#             connection.execute(text("""
#                 ALTER TABLE text_entries 
#                 ADD COLUMN IF NOT EXISTS is_human_translation BOOLEAN DEFAULT FALSE
#             """))
#             connection.commit()
#         except Exception as e:
#             print(f"Migration error: {str(e)}")

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)