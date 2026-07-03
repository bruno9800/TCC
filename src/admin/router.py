"""
Admin Router — Autenticação e Gerenciamento de Documentos

Todas as rotas exigem um AdminUser autenticado via Bearer JWT (get_current_admin),
exceto o próprio login. Autenticação deliberadamente separada da x-api-key
pública que protege /chat, /documents e /logs (ver src/auth.py).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from src.admin.auth import create_access_token, get_current_admin, verify_password
from src.admin.schemas import (
    DocumentOut,
    DocumentUpdateRequest,
    LoginRequest,
    TokenResponse,
)
from src.db.models import AdminUser
from src.db.session import get_db
from src.documents import service as document_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Autenticação ───────────────────────────────────────────────────────────────


@router.post("/auth/login", response_model=TokenResponse, summary="Login de administrador")
async def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    admin = db.query(AdminUser).filter_by(email=payload.email).one_or_none()
    if admin is None or not verify_password(payload.password, admin.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="E-mail ou senha inválidos",
        )
    return TokenResponse(access_token=create_access_token(admin))


# ── Documentos ─────────────────────────────────────────────────────────────────


@router.post(
    "/documents",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload de um novo documento (dispara ETL → chunk → embed → index)",
)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base_slug: str = Form(...),
    course_id: int | None = Form(None),
    title: str | None = Form(None),
    category: str | None = Form(None),
    admin: AdminUser = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> DocumentOut:
    file_bytes = await file.read()
    try:
        document = document_service.create_document(
            db,
            file_bytes=file_bytes,
            filename=file.filename or "documento.pdf",
            knowledge_base_slug=knowledge_base_slug,
            course_id=course_id,
            title=title,
            category=category,
            admin_id=admin.id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return document


@router.get("/documents", response_model=list[DocumentOut], summary="Lista documentos")
async def list_documents(
    course_id: int | None = None,
    knowledge_base_id: int | None = None,
    status_filter: str | None = None,
    admin: AdminUser = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[DocumentOut]:
    return document_service.list_documents(
        db, course_id=course_id, knowledge_base_id=knowledge_base_id, status=status_filter
    )


@router.get("/documents/{document_id}", response_model=DocumentOut, summary="Detalhe de um documento")
async def get_document(
    document_id: int,
    admin: AdminUser = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> DocumentOut:
    document = document_service.get_document(db, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Documento não encontrado")
    return document


@router.patch("/documents/{document_id}", response_model=DocumentOut, summary="Atualiza metadados")
async def update_document(
    document_id: int,
    payload: DocumentUpdateRequest,
    admin: AdminUser = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> DocumentOut:
    try:
        return document_service.update_document(
            db, document_id, **payload.model_dump(exclude_unset=True)
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove um documento e seus vetores",
)
async def delete_document(
    document_id: int,
    admin: AdminUser = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> None:
    try:
        document_service.delete_document(db, document_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/documents/{document_id}/reindex",
    response_model=DocumentOut,
    summary="Força reprocessamento (ETL → chunk → embed → index)",
)
async def reindex_document(
    document_id: int,
    admin: AdminUser = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> DocumentOut:
    try:
        return document_service.reindex_document(db, document_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
