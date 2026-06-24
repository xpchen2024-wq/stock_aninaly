# ============================================================================
# AI Stock Analysis Platform - Auth API
# ============================================================================
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import jwt

from app.config import get_settings
from app.database import get_db
from app.models import User

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()
security = HTTPBearer()

# Use bcrypt directly (passlib has compat issues with bcrypt 4.x)
import bcrypt


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False


# -- Schemas --
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class UserInfo(BaseModel):
    id: str
    username: str
    email: str
    is_superuser: bool


# -- Helpers --
def create_token(user_id: str, username: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "username": username,
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    try:
        payload = decode_token(credentials.credentials)
        user_id = payload.get("sub")
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# -- Routes --
@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    logger.info(f"Register request: username={req.username}, email={req.email}")
    existing = await db.execute(select(User).where(
        (User.username == req.username) | (User.email == req.email)
    ))
    if existing.scalar_one_or_none():
        logger.warning(f"Register failed: username or email already exists ({req.username})")
        raise HTTPException(status_code=400, detail="Username or email already exists")

    user = User(
        username=req.username,
        email=req.email,
        hashed_password=hash_password(req.password),
    )
    db.add(user)
    await db.flush()

    token = create_token(str(user.id), user.username)
    logger.info(f"User registered: id={user.id}, username={user.username}")
    return TokenResponse(
        access_token=token,
        expires_in=settings.JWT_EXPIRE_MINUTES * 60,
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    logger.info(f"Login request: username={req.username}")
    result = await db.execute(select(User).where(User.username == req.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(req.password, user.hashed_password):
        logger.warning(f"Login failed: invalid credentials for username={req.username}")
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not user.is_active:
        logger.warning(f"Login failed: account disabled for username={req.username}")
        raise HTTPException(status_code=403, detail="Account disabled")

    token = create_token(str(user.id), user.username)
    logger.info(f"Login success: user_id={user.id}, username={user.username}")
    return TokenResponse(
        access_token=token,
        expires_in=settings.JWT_EXPIRE_MINUTES * 60,
    )


@router.get("/me", response_model=UserInfo)
async def get_me(user: User = Depends(get_current_user)):
    logger.debug(f"Get current user: id={user.id}, username={user.username}")
    return UserInfo(
        id=str(user.id),
        username=user.username,
        email=user.email,
        is_superuser=user.is_superuser,
    )
