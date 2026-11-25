"""FastAPI layer - schemas and router.

Deliberately no `from api.router import router` re-export here: it would
rebind the `api.router` attribute to the APIRouter instance, so
`import api.router as m` and `patch("api.router.<name>")` would silently
reach the router object instead of the module. Import
`from api.router import router` directly.
"""
