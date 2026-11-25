"""Wiring: session -> user db -> user manager -> `FastAPIUsers` instance.

`current_active_user` is the dependency every user-owned endpoint in
`api/router.py` requires (`Depends(current_active_user)`) instead of the
old `ensure_default_user()` call - see docs/DEVELOPMENT.md's pipeline notes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends
from fastapi_users import FastAPIUsers
from fastapi_users.db import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from auth.backend import auth_backend
from auth.manager import UserManager
from database.db import get_session
from database.models import User


async def get_async_session() -> AsyncIterator[AsyncSession]:
    async with get_session() as session:
        yield session


async def get_user_db(
    session: AsyncSession = Depends(get_async_session),
) -> AsyncIterator[SQLAlchemyUserDatabase]:
    yield SQLAlchemyUserDatabase(session, User)


async def get_user_manager(
    user_db: SQLAlchemyUserDatabase = Depends(get_user_db),
) -> AsyncIterator[UserManager]:
    yield UserManager(user_db)


fastapi_users = FastAPIUsers[User, str](get_user_manager, [auth_backend])

current_active_user = fastapi_users.current_user(active=True)
current_superuser = fastapi_users.current_user(active=True, superuser=True)
