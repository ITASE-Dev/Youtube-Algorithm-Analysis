"""MSSQL engine, session factory and session helpers.

Every service imports its database access from here so connection pooling,
retry behaviour and transaction semantics stay identical across the monorepo.

Typical usage
-------------
Script / worker code (auto commit-or-rollback)::

    from core.database import session_scope

    with session_scope() as db:
        db.add(video)

FastAPI-style dependency (caller owns the commit)::

    from core.database import get_db

    def endpoint(db: Session = Depends(get_db)):
        ...

The engine is created lazily on first use so that merely importing this module
(e.g. during `--help` or unit-test collection) never opens a socket.
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from core.config import ConfigurationError, settings
from core.models import Base

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker[Session]] = None


class DatabaseConnectionError(RuntimeError):
    """Raised when the MSSQL engine cannot be created or reached."""


def _create_engine() -> Engine:
    """Build the SQLAlchemy engine for MSSQL over pyodbc.

    Raises:
        DatabaseConnectionError: If the driver is unavailable or the URL is invalid.
    """
    try:
        import pyodbc  # noqa: F401  (imported for the clearer error message below)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DatabaseConnectionError(
            "pyodbc is not installed. Run `pip install pyodbc` and make sure the "
            "Microsoft ODBC Driver for SQL Server is installed on this machine."
        ) from exc

    try:
        url = settings.sqlalchemy_database_uri
    except ConfigurationError as exc:
        raise DatabaseConnectionError(str(exc)) from exc

    logger.info("Creating SQLAlchemy engine for %s", settings.masked_summary())

    try:
        engine = create_engine(
            url,
            echo=settings.db_echo,
            pool_pre_ping=True,          # transparently drops stale/killed connections
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
            future=True,
            # Batches multi-row INSERTs; a large win for bulk video ingestion.
            fast_executemany=True,
        )
    except SQLAlchemyError as exc:
        raise DatabaseConnectionError(f"Could not create the MSSQL engine: {exc}") from exc

    @event.listens_for(engine, "connect")
    def _set_session_options(dbapi_connection, _connection_record):  # pragma: no cover
        """Fail fast and consistently rather than leaving half-applied writes."""
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("SET ARITHABORT ON; SET XACT_ABORT ON;")
        finally:
            cursor.close()

    return engine


def get_engine() -> Engine:
    """Return the process-wide engine, creating it on first call."""
    global _engine
    if _engine is None:
        _engine = _create_engine()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide configured :class:`sessionmaker`."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,  # attributes stay usable after commit
            future=True,
        )
    return _SessionFactory


# Direct handle for callers that want `SessionLocal()` semantics.
def SessionLocal(**kwargs) -> Session:  # noqa: N802 - conventional SQLAlchemy name
    """Create a new :class:`~sqlalchemy.orm.Session`. Caller must close it."""
    return get_session_factory()(**kwargs)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commits on success, rolls back on any exception.

    Example::

        with session_scope() as db:
            db.add(Channel(channel_id="UC..."))
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Transaction rolled back")
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """Dependency-injection style session provider (FastAPI ``Depends`` friendly).

    Yields a session and always closes it. The caller decides when to commit;
    any in-flight transaction is rolled back if the caller raised.
    """
    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(*, checkfirst: bool = True) -> None:
    """Create every table declared on :class:`core.models.Base`.

    Safe to call repeatedly; existing tables are left untouched. For real schema
    evolution prefer Alembic migrations over this helper.
    """
    engine = get_engine()
    logger.info("Creating tables: %s", ", ".join(sorted(Base.metadata.tables)))
    Base.metadata.create_all(bind=engine, checkfirst=checkfirst)
    logger.info("Schema is up to date.")


def healthcheck() -> bool:
    """Return True when a trivial round-trip to MSSQL succeeds."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError as exc:
        logger.error("Database healthcheck failed: %s", exc)
        return False


def dispose_engine() -> None:
    """Close all pooled connections. Call before forking or on shutdown."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
        logger.info("Engine disposed.")
    _engine = None
    _SessionFactory = None


if __name__ == "__main__":  # pragma: no cover - manual bootstrap helper
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not healthcheck():
        raise SystemExit("Cannot reach the database - check your .env settings.")
    init_db()
    print(f"OK: schema ready on {settings.masked_summary()}")


__all__ = [
    "DatabaseConnectionError",
    "SessionLocal",
    "dispose_engine",
    "get_db",
    "get_engine",
    "get_session_factory",
    "healthcheck",
    "init_db",
    "session_scope",
]
