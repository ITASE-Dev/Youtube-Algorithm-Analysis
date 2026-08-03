"""Shared core package: configuration, database access and ORM models.

Imported by every service in the monorepo::

    from core.config import settings
    from core.database import session_scope
    from core.models import Channel, Video
"""

from core.config import Settings, get_settings, settings
from core.database import (
    SessionLocal,
    dispose_engine,
    get_db,
    get_engine,
    healthcheck,
    init_db,
    session_scope,
)
from core.models import Base, Channel, Video

__version__ = "0.1.0"

__all__ = [
    "Base",
    "Channel",
    "SessionLocal",
    "Settings",
    "Video",
    "__version__",
    "dispose_engine",
    "get_db",
    "get_engine",
    "get_settings",
    "healthcheck",
    "init_db",
    "session_scope",
    "settings",
]
