"""Authentication.

Built on FastAPI-Users. `auth/users.py` wires up `fastapi_users`'s register/login/me
routers and the `current_active_user` dependency every protected endpoint
in `api/router.py` requires. See `auth/manager.py` for the one real
customization (a string-UUID `parse_id`, since `database.models.User`
uses `id: str`, not fastapi-users' default UUID type).
"""
