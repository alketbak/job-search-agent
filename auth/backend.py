"""JWT bearer auth backend - `POST /auth/jwt/login` issues a token,
subsequent requests send it as `Authorization: Bearer <token>`.
"""

from __future__ import annotations

from fastapi_users.authentication import AuthenticationBackend, BearerTransport, JWTStrategy

from config import settings

bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy:
    if not settings.auth_secret:
        raise RuntimeError("AUTH_SECRET is not set - required for authentication.")
    return JWTStrategy(secret=settings.auth_secret, lifetime_seconds=settings.auth_token_lifetime_seconds)


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)
