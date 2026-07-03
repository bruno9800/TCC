"""
Autenticação de Administrador (JWT)

Separada da x-api-key pública (ver src/auth.py) que protege /chat, /documents
e /logs. Rotas administrativas exigem um AdminUser autenticado via Bearer JWT.

Espelha o padrão Security/Depends já usado em src/auth.py — mesma ideia,
esquema diferente (Bearer JWT em vez de header de API key estática).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from src.config import JWT_ALGORITHM, JWT_EXPIRE_MINUTES, JWT_SECRET
from src.db.models import AdminUser
from src.db.session import get_db

_bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Gera o hash bcrypt de uma senha em texto plano."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verifica uma senha em texto plano contra um hash bcrypt."""
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(admin: AdminUser) -> str:
    """Emite um JWT assinado para o AdminUser, válido por JWT_EXPIRE_MINUTES."""
    payload = {
        "sub": str(admin.id),
        "email": admin.email,
        "role": admin.role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> AdminUser:
    """
    Dependência FastAPI que valida o token Bearer e retorna o AdminUser.

    Usada em todas as rotas de /admin/* (exceto o próprio login).
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de administrador ausente",
        )

    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido ou expirado",
        )

    admin = db.get(AdminUser, int(payload["sub"]))
    if admin is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Administrador não encontrado",
        )
    return admin
