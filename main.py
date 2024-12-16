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


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Database Configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgres://u9vtpvl0p3bo57:p8a3394ca003ac09a6d33d10136b1a9322e1e88e9d5837b89ece29192dbc0d2e5@cf980tnnkgv1bp.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/dblv81pilc1dsu"
    #"postgresql://postgres:Aj0jwttg88!@localhost:5432/trans25"
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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    print('Request for index page received')
    return templates.TemplateResponse('index.html', {"request": request})

@app.get('/favicon.ico')
async def favicon():
    file_name = 'favicon.ico'
    file_path = './static/' + file_name
    return FileResponse(path=file_path, headers={'mimetype': 'image/vnd.microsoft.icon'})

@app.post('/hello', response_class=HTMLResponse)
async def hello(request: Request, name: str = Form(...)):
    if name:
        print('Request for hello page received with name=%s' % name)
        return templates.TemplateResponse('hello.html', {"request": request, 'name':name})
    else:
        print('Request for hello page received with no name or blank name -- redirecting')
        return RedirectResponse(request.url_for("index"), status_code=status.HTTP_302_FOUND)

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

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)

