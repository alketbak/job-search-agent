"""The fastapi-users `UserManager` - password hashing, token minting, and
the lifecycle hooks (`on_after_register`, etc.) all go through this.

Two real customizations beyond the library's default:
1. `parse_id`: `database.models.User.id` is a uuid4-as-string (matching
   every other table in this app), not fastapi-users' default UUID type.
2. The reset-password/verification token secrets are set in `__init__`,
   not as class attributes, so a missing `AUTH_SECRET` raises when a
   manager is actually constructed (per-request, via `get_user_manager`)
   rather than at import time - importing this module must stay safe even
   when auth isn't configured (e.g. most of the existing test suite).
"""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi_users import BaseUserManager
from fastapi_users.db import SQLAlchemyUserDatabase

from config import settings
from database.models import User

log = logging.getLogger(__name__)


class UserManager(BaseUserManager[User, str]):
    def __init__(self, user_db: SQLAlchemyUserDatabase) -> None:
        if not settings.auth_secret:
            raise RuntimeError("AUTH_SECRET is not set - required for authentication.")
        self.reset_password_token_secret = settings.auth_secret
        self.verification_token_secret = settings.auth_secret
        super().__init__(user_db)

    def parse_id(self, value: object) -> str:
        return str(value)

    async def on_after_register(self, user: User, request: Request | None = None) -> None:
        log.info("User registered: %s", user.id)
