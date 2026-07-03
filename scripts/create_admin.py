#!/usr/bin/env python3
"""
Script de bootstrap — cria um AdminUser.

Uso:
    python scripts/create_admin.py <email> <senha> [--role admin|editor]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.admin.auth import hash_password
from src.db.models import AdminUser
from src.db.session import SessionLocal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("email")
    parser.add_argument("password")
    parser.add_argument("--role", default="admin", choices=["admin", "editor"])
    args = parser.parse_args()

    db = SessionLocal()
    try:
        existing = db.query(AdminUser).filter_by(email=args.email).one_or_none()
        if existing:
            print(f"Já existe um AdminUser com o e-mail {args.email}")
            return

        admin = AdminUser(
            email=args.email,
            password_hash=hash_password(args.password),
            role=args.role,
        )
        db.add(admin)
        db.commit()
        print(f"AdminUser criado: {admin.email} ({admin.role})")
    finally:
        db.close()


if __name__ == "__main__":
    main()
