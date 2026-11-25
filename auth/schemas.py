"""Pydantic request/response schemas for fastapi-users' routers."""

from __future__ import annotations

from fastapi_users import schemas


class UserRead(schemas.BaseUser[str]):
    is_default: bool


class UserCreate(schemas.BaseUserCreate):
    pass


class UserUpdate(schemas.BaseUserUpdate):
    pass
