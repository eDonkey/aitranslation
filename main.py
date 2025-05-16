from fastapi import FastAPI, Header, HTTPException, Depends, Request, Form, Path, status, Query
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, text, Boolean, DateTime, func, Float, ForeignKey, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
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
import logging
from jinja2 import Environment
from markupsafe import escape
from alembic import op
import sqlalchemy as sa

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
    subscriptions = relationship("UserSubscription", back_populates="user")

class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)  # AI ONLY, HUMAN ONLY, FULL PACKAGE
    price = Column(Float, nullable=False)  # Price of the subscription
    description = Column(String, nullable=True)  # Optional description
    created_at = Column(DateTime, default=datetime.utcnow)

class UserSubscription(Base):
    __tablename__ = "user_subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id"), nullable=False)
    is_active = Column(Boolean, default=True)
    start_date = Column(DateTime, default=datetime.utcnow)
    renew_date = Column(DateTime, nullable=True)  # Next renewal date
    end_date = Column(DateTime, nullable=True)  # Optional for expiration
    user = relationship("User", back_populates="subscriptions")
    subscription = relationship("Subscription")

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

# Helper function to get the current active admin user
async def get_current_active_admin(
    request: Request
) -> User:
    current_user = getattr(request.state, "user", None)
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    if not current_user.is_active:
        raise HTTPException(status_code=403, detail="Inactive user")
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not an admin")
    return current_user

@app.put("/api/admin/api-keys/{key_id}/toggle")
async def toggle_api_key_status(
    key_id: int,
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    api_key = db.query(APIKey).filter(APIKey.id == key_id).first()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    # Toggle the active status
    api_key.is_active = not api_key.is_active
    db.commit()
    return {"message": f"API key {'activated' if api_key.is_active else 'deactivated'} successfully"}

@app.get("/api/admin/api-keys/{key_id}/stats")
async def get_api_key_stats(
    key_id: int,
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    api_key = db.query(APIKey).filter(APIKey.id == key_id).first()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    # Fetch stats for the API key
    usage_count = db.query(TextEntry).filter(TextEntry.apikey_requested == api_key.key).count()
    last_used = api_key.last_used  # Ensure this field exists in your database model
    return {
        "name": api_key.name,
        "usage_count": usage_count,
        "last_used": last_used.isoformat() if last_used else None,
        "is_active": api_key.is_active
    }

openai.api_key = os.getenv("OPENAI_API_KEY")

templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

def escapejs(value):
    return escape(value).replace("'", "\\'").replace('"', '\\"')

# Add the filter to your Jinja2 environment
templates.env.filters['escapejs'] = escapejs

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("traductor.html", {
        "request": request,
        # Add any additional context data here if needed
    })

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    # Check if the username or email already exists
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already exists")
    
    # Create the user
    hashed_password = get_password_hash(password)
    new_user = User(username=username, email=email, hashed_password=hashed_password, is_active=True)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Redirect to subscription selection
    return RedirectResponse(url=f"/select-subscription?user_id={new_user.id}", status_code=303)

@app.get("/select-subscription", response_class=HTMLResponse)
async def select_subscription_page(user_id: int, request: Request, db: Session = Depends(get_db)):
    subscriptions = db.query(Subscription).all()
    return templates.TemplateResponse("select_subscription.html", {
        "request": request,
        "user_id": user_id,
        "subscriptions": subscriptions
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

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.username == username).first()
    if user and verify_password(password, user.hashed_password):
        return user
    return None

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
        raise HTTPException(status_code=403, detail="Inactive user")
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not an admin")
    return current_user

# Authentication routes
@app.post("/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    logger.info(f"Login attempt for username: {form_data.username}")
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        logger.info("Invalid login attempt.")
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate token
    access_token = create_access_token(data={"sub": user.username, "is_admin": user.is_admin})
    logger.info(f"Generated token for user {user.username}: {access_token}")
    
    # Set token in cookie
    response = RedirectResponse(
        url="/admin/dashboard" if user.is_admin else "/client/dashboard",
        status_code=303
    )
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax",
        max_age=1800  # 30 minutes
    )
    return response

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/logout")
async def logout(request: Request):
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("access_token")
    return response

@app.post("/logout")
async def logout(request: Request):
    """
    Logs out the current user by clearing the authentication cookie.
    """
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(key="access_token", httponly=True, secure=True, samesite="lax")
    return response

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    logger.info(f"Accessing admin dashboard with user: {current_user.username}")
    
 # Current time
    now = datetime.utcnow()

    # Time ranges
    last_24_hours = now - timedelta(hours=24)
    last_7_days = now - timedelta(days=7)
    last_30_days = now - timedelta(days=30)

    # Manual translations
    manual_translations_last_24_hours = db.query(TextEntry).filter(
        TextEntry.is_human_translation == True,
        TextEntry.created_at >= last_24_hours
    ).count()

    manual_translations_last_7_days = db.query(TextEntry).filter(
        TextEntry.is_human_translation == True,
        TextEntry.created_at >= last_7_days
    ).count()

    manual_translations_last_30_days = db.query(TextEntry).filter(
        TextEntry.is_human_translation == True,
        TextEntry.created_at >= last_30_days
    ).count()

    # AI translations
    ai_translations_last_24_hours = db.query(TextEntry).filter(
        TextEntry.is_human_translation == False,
        TextEntry.created_at >= last_24_hours
    ).count()

    ai_translations_last_7_days = db.query(TextEntry).filter(
        TextEntry.is_human_translation == False,
        TextEntry.created_at >= last_7_days
    ).count()

    ai_translations_last_30_days = db.query(TextEntry).filter(
        TextEntry.is_human_translation == False,
        TextEntry.created_at >= last_30_days
    ).count()
    # Top 10 API keys by usage
    top_10_api_keys = db.query(
        TextEntry.apikey_requested,
        func.count(TextEntry.id).label("usage_count")
    ).group_by(TextEntry.apikey_requested).order_by(desc("usage_count")).limit(10).all()

    # Prepare stats object
    stats = {
        "manual_translations": {
            "last_24_hours": manual_translations_last_24_hours,
            "last_7_days": manual_translations_last_7_days,
            "last_30_days": manual_translations_last_30_days,
        },
        "ai_translations": {
            "last_24_hours": ai_translations_last_24_hours,
            "last_7_days": ai_translations_last_7_days,
            "last_30_days": ai_translations_last_30_days,
        },
        "top_10_api_keys": [(key, count) for key, count in top_10_api_keys]
    }

    
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "user": current_user,
        "stats": stats  # Pass stats to the template
    })

@app.get("/admin/subscriptions", response_class=HTMLResponse)
async def admin_subscriptions(
    request: Request,
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    subscriptions = db.query(Subscription).all()
    return templates.TemplateResponse("admin_subscriptions.html", {
        "request": request,
        "subscriptions": subscriptions
    })

@app.post("/admin/subscriptions")
async def create_or_update_subscription(
    id: Optional[int] = Form(None),  # Make `id` optional
    name: str = Form(...),
    price: float = Form(...),
    description: str = Form(None),
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    if id:
        # Update existing subscription
        subscription = db.query(Subscription).filter(Subscription.id == id).first()
        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")
        subscription.name = name
        subscription.price = price
        subscription.description = description
    else:
        # Create new subscription
        subscription = Subscription(name=name, price=price, description=description)
        db.add(subscription)
    db.commit()
    return RedirectResponse(url="/admin/subscriptions", status_code=303)

@app.post("/subscribe/{subscription_id}")
async def subscribe_to_plan(
    subscription_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    subscription = db.query(Subscription).filter(Subscription.id == subscription_id).first()
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    # Create or update user subscription
    user_subscription = db.query(UserSubscription).filter(
        UserSubscription.user_id == current_user.id,
        UserSubscription.subscription_id == subscription_id
    ).first()
    if user_subscription:
        user_subscription.is_active = True
        user_subscription.start_date = datetime.utcnow()
        user_subscription.renew_date = datetime.utcnow() + timedelta(days=30)
    else:
        user_subscription = UserSubscription(
            user_id=current_user.id,
            subscription_id=subscription_id,
            is_active=True,
            start_date=datetime.utcnow(),
            renew_date=datetime.utcnow() + timedelta(days=30)
        )
        db.add(user_subscription)
    db.commit()
    return {"message": f"Subscribed to {subscription.name} successfully"}

@app.post("/renew/{subscription_id}")
async def renew_subscription(
    subscription_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user_subscription = db.query(UserSubscription).filter(
        UserSubscription.user_id == current_user.id,
        UserSubscription.subscription_id == subscription_id,
        UserSubscription.is_active == True
    ).first()
    if not user_subscription:
        raise HTTPException(status_code=404, detail="Active subscription not found")
    
    # Update the renew date
    user_subscription.renew_date = user_subscription.renew_date + timedelta(days=30)
    db.commit()
    return {"message": f"Subscription renewed successfully. Next renewal date: {user_subscription.renew_date}"}

@app.post("/payment/simulate")
async def simulate_payment(
    subscription_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    subscription = db.query(Subscription).filter(Subscription.id == subscription_id).first()
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    # Simulate payment success
    return {"message": f"Payment for {subscription.name} (${subscription.price}) successful"}

@app.post("/simulate-payment")
async def simulate_payment(
    user_id: int = Form(...),
    subscription_id: int = Form(...),
    db: Session = Depends(get_db)
):
    # Fetch the subscription
    subscription = db.query(Subscription).filter(Subscription.id == subscription_id).first()
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    # Simulate payment success
    user_subscription = UserSubscription(
        user_id=user_id,
        subscription_id=subscription_id,
        is_active=True,
        start_date=datetime.utcnow(),
        renew_date=datetime.utcnow() + timedelta(days=30)
    )
    db.add(user_subscription)
    db.commit()
    
    return RedirectResponse(url="/my-subscriptions", status_code=303)

@app.get("/my-subscriptions", response_class=HTMLResponse)
async def my_subscriptions(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    subscriptions = db.query(UserSubscription).filter(
        UserSubscription.user_id == current_user.id,
        UserSubscription.is_active == True
    ).all()
    return templates.TemplateResponse("my_subscriptions.html", {
        "request": request,
        "subscriptions": subscriptions
    })

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
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db)
):
    api_keys = db.query(APIKey).all()
    return templates.TemplateResponse("admin_keys.html", {
        "request": request,
        "api_keys": api_keys
    })

@app.get("/client/dashboard", response_class=HTMLResponse)
async def client_dashboard(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Accessing client dashboard with user: {current_user.username}")
    
    # Redirect admin users to the admin dashboard
    if current_user.is_admin:
        logger.info(f"User {current_user.username} is an admin. Redirecting to admin dashboard.")
        return RedirectResponse(url="/admin/dashboard", status_code=303)
    
    # Fetch the last 10 human translations
    human_translations = db.query(TextEntry)\
        .filter(TextEntry.apikey_requested == current_user.api_key, TextEntry.is_human_translation == True)\
        .order_by(TextEntry.created_at.desc())\
        .limit(10)\
        .all()

    # Fetch the last 10 AI translations
    ai_translations = db.query(TextEntry)\
        .filter(TextEntry.apikey_requested == current_user.api_key, TextEntry.is_human_translation == False)\
        .order_by(TextEntry.created_at.desc())\
        .limit(10)\
        .all()

    # Prepare translation data
    def prepare_translation_data(translations):
        return [
            {
                "id": translation.id,
                "created_at": translation.created_at,
                "status": "Completed" if all(
                    getattr(translation, lang) is not None for lang in ["english", "spanish", "portuguese", "french", "deutch", "italian"]
                ) else "Pending",
                "original_text": next(
                    (getattr(translation, lang) for lang in ["english", "spanish", "portuguese", "french", "deutch", "italian"] if getattr(translation, lang) is not None),
                    None
                )
            }
            for translation in translations
        ]

    return templates.TemplateResponse("client_dashboard.html", {
        "request": request,
        "user": current_user,
        "translations": {
            "human": prepare_translation_data(human_translations),
            "ai": prepare_translation_data(ai_translations)
        }
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

@app.middleware("http")
async def authenticate_user_middleware(request: Request, call_next):
    # Skip authentication for login and static routes
    if request.url.path in ["/", "/login", "/register", "/token"] or request.url.path.startswith("/static"):
        return await call_next(request)

    # Get token from header or cookie
    token = None
    auth_cookie = request.cookies.get('access_token')
    if auth_cookie and auth_cookie.startswith('Bearer '):
        token = auth_cookie.split(' ')[1]
        logger.info(f"Token found in cookie: {token}")
    else:
        logger.info("No token found. Redirecting to login.")
        return RedirectResponse(url="/login", status_code=303)

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        logger.info(f"Decoded token payload: {payload}")
        if username:
            db = SessionLocal()
            try:
                user = db.query(User).filter(User.username == username).first()
                if user and user.is_active:
                    request.state.user = user
                    response = await call_next(request)
                    return response
            finally:
                db.close()
    except JWTError as e:
        logger.error(f"JWT decoding error: {str(e)}")

    return RedirectResponse(url="/login", status_code=303)

@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users_page(
    request: Request,
    current_user: User = Depends(get_current_active_admin),
    db: Session = Depends(get_db),
    page: int = Query(1, gt=0),  # Default to page 1
    search: str = Query(None)   # Optional search query
):
    query = db.query(User)
    
    # Apply search filter if provided
    if search:
        query = query.filter(
            (User.username.ilike(f"%{search}%")) | (User.email.ilike(f"%{search}%"))
        )
    
    # Pagination
    total_users = query.count()
    users = query.offset((page - 1) * 25).limit(25).all()
    
    return templates.TemplateResponse(
        "admin_users.html",
        {
            "request": request,
            "users": users,
            "total_users": total_users,
            "current_page": page,
            "search_query": search,
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

@app.put("/api/users/change-password")
async def change_password(
    current_password: str = Form(...),
    new_password: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Endpoint to allow users to change their password.
    """
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="User not authenticated"
        )
    
    # Verify the current password
    if not verify_password(current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=400,
            detail="Current password is incorrect"
        )
    
    # Hash the new password and update the user record
    current_user.hashed_password = get_password_hash(new_password)
    db.commit()
    
    return {"message": "Password updated successfully"}

@app.post("/api/users/forgot-password")
async def forgot_password(
    email: str = Form(...),
    new_password: str = Form(...),
    reset_token: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Endpoint to reset a user's password using a reset token.
    """
    # Find the user by email
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User with this email does not exist"
        )
    
    # Verify the reset token (this assumes you have a token verification mechanism)
    try:
        payload = jwt.decode(reset_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("sub") != user.email:
            raise HTTPException(
                status_code=400,
                detail="Invalid reset token"
            )
    except JWTError:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired reset token"
        )
    
    # Hash the new password and update the user record
    user.hashed_password = get_password_hash(new_password)
    db.commit()
    
    return {"message": "Password reset successfully"}

@app.post("/api/users/request-password-reset")
async def request_password_reset(
    email: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Endpoint to request a password reset token.
    """
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User with this email does not exist"
        )
    
    # Generate a reset token
    reset_token = create_access_token(data={"sub": user.email})
    
    # Send the token via email (implement your email-sending logic here)
    # Example: send_email(user.email, "Password Reset", f"Your reset token: {reset_token}")
    
    return {"message": f"Password reset token generated: {reset_token}"}

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

# Add renew_date column
def upgrade():
    op.add_column('user_subscriptions', sa.Column('renew_date', sa.DateTime(), nullable=True))

def downgrade():
    op.drop_column('user_subscriptions', 'renew_date')

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)