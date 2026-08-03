"""Central configuration for the YouTube Video Performance Prediction Analyzer.

Every service in the monorepo (collector, AI analyzer, predictor engine) imports
its configuration from here so that credentials and tunables live in exactly one
place: the ``.env`` file at the workspace root.

Usage
-----
    from core.config import settings

    print(settings.sqlalchemy_database_uri)
"""

from __future__ import annotations

import os
import re
import urllib.parse
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Workspace root == parent of the `core` package. The .env lives next to it.
BASE_DIR: Path = Path(__file__).resolve().parent.parent
ENV_PATH: Path = BASE_DIR / ".env"

# `override=False` keeps real environment variables (CI / container secrets)
# authoritative over whatever happens to sit in a local .env file.
load_dotenv(dotenv_path=ENV_PATH, override=False)


class ConfigurationError(RuntimeError):
    """Raised when a required environment variable is missing or malformed."""


def _get(key: str, default: str | None = None, *, required: bool = False) -> str:
    """Read an environment variable with optional strictness.

    Args:
        key: Environment variable name.
        default: Fallback used when the variable is absent or empty.
        required: When True, raise instead of falling back.

    Raises:
        ConfigurationError: If ``required`` and no value could be resolved.
    """
    value = os.getenv(key, "").strip() or (default or "")
    if not value and required:
        raise ConfigurationError(
            f"Missing required environment variable '{key}'. "
            f"Add it to {ENV_PATH} or export it in your shell."
        )
    return value


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"Environment variable '{key}' must be an integer, got {raw!r}.") from exc


def _get_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable, validated view over the process environment."""

    # --- MSSQL -----------------------------------------------------------
    mssql_server: str = field(default_factory=lambda: _get("MSSQL_SERVER", "localhost"))
    mssql_port: int = field(default_factory=lambda: _get_int("MSSQL_PORT", 1433))
    mssql_database: str = field(default_factory=lambda: _get("MSSQL_DATABASE", "YoutubeAnalysis"))
    mssql_user: str = field(default_factory=lambda: _get("MSSQL_USER", ""))
    mssql_password: str = field(default_factory=lambda: _get("MSSQL_PASSWORD", ""))
    mssql_driver: str = field(default_factory=lambda: _get("MSSQL_DRIVER", "ODBC Driver 17 for SQL Server"))
    mssql_trusted_connection: bool = field(default_factory=lambda: _get_bool("MSSQL_TRUSTED_CONNECTION", False))
    mssql_encrypt: bool = field(default_factory=lambda: _get_bool("MSSQL_ENCRYPT", True))
    mssql_trust_server_certificate: bool = field(
        default_factory=lambda: _get_bool("MSSQL_TRUST_SERVER_CERTIFICATE", True)
    )

    # Escape hatches, in order of precedence:
    #   1. DATABASE_URL  -- a complete SQLAlchemy URL
    #   2. DB_CONN_STR   -- a raw ODBC connection string (handles named
    #                       instances like HOST\SQLEXPRESS, which the
    #                       SERVER,PORT form above cannot express)
    #   3. the MSSQL_* fields
    database_url: str = field(default_factory=lambda: _get("DATABASE_URL", ""))
    db_conn_str: str = field(default_factory=lambda: _get("DB_CONN_STR", ""))

    # --- Engine tunables -------------------------------------------------
    db_echo: bool = field(default_factory=lambda: _get_bool("DB_ECHO", False))
    db_pool_size: int = field(default_factory=lambda: _get_int("DB_POOL_SIZE", 5))
    db_max_overflow: int = field(default_factory=lambda: _get_int("DB_MAX_OVERFLOW", 10))
    db_pool_timeout: int = field(default_factory=lambda: _get_int("DB_POOL_TIMEOUT", 30))
    db_pool_recycle: int = field(default_factory=lambda: _get_int("DB_POOL_RECYCLE", 1800))

    # --- External APIs ---------------------------------------------------
    youtube_api_key: str = field(default_factory=lambda: _get("YOUTUBE_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: _get("OPENAI_API_KEY", ""))
    openai_text_model: str = field(default_factory=lambda: _get("OPENAI_TEXT_MODEL", "gpt-4o-mini"))
    openai_vision_model: str = field(default_factory=lambda: _get("OPENAI_VISION_MODEL", "gpt-4o"))
    returnyoutubedislike_api_url: str = field(
        default_factory=lambda: _get(
            "RETURNYOUTUBEDISLIKE_API_URL", "https://returnyoutubedislikeapi.com/votes"
        )
    )

    # --- Misc ------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO").upper())
    media_cache_dir: Path = field(
        default_factory=lambda: Path(_get("MEDIA_CACHE_DIR", str(BASE_DIR / ".media_cache")))
    )

    @property
    def odbc_connection_string(self) -> str:
        """Return the raw ODBC connection string handed to pyodbc.

        Uses ``DB_CONN_STR`` verbatim when set; otherwise assembles one from the
        individual ``MSSQL_*`` variables.
        """
        if self.db_conn_str:
            return self.db_conn_str

        parts = [
            f"DRIVER={{{self.mssql_driver}}}",
            f"SERVER={self.mssql_server},{self.mssql_port}",
            f"DATABASE={self.mssql_database}",
        ]
        if self.mssql_trusted_connection:
            parts.append("Trusted_Connection=yes")
        else:
            if not self.mssql_user or not self.mssql_password:
                raise ConfigurationError(
                    "MSSQL_USER and MSSQL_PASSWORD are required unless "
                    "MSSQL_TRUSTED_CONNECTION=true (Windows authentication)."
                )
            parts.append(f"UID={self.mssql_user}")
            parts.append(f"PWD={self.mssql_password}")

        parts.append(f"Encrypt={'yes' if self.mssql_encrypt else 'no'}")
        parts.append(f"TrustServerCertificate={'yes' if self.mssql_trust_server_certificate else 'no'}")
        return ";".join(parts)

    @property
    def sqlalchemy_database_uri(self) -> str:
        """SQLAlchemy URL for the MSSQL/pyodbc dialect.

        Passing the whole ODBC string through ``odbc_connect`` avoids the
        classic escaping headaches with passwords containing ``@``, ``:`` or
        ``/``.
        """
        if self.database_url:
            return self.database_url
        encoded = urllib.parse.quote_plus(self.odbc_connection_string)
        return f"mssql+pyodbc:///?odbc_connect={encoded}"

    def masked_summary(self) -> str:
        """Safe-to-log description of the active connection target.

        Never includes a password: ``PWD``/``Password`` values are redacted.
        """
        if self.database_url:
            # Strip any user:password@ segment before logging.
            return f"MSSQL via DATABASE_URL ({re.sub(r'//[^@/]*@', '//***@', self.database_url)})"

        if self.db_conn_str:
            kv = self._parse_odbc(self.db_conn_str)
            auth = "Windows auth" if kv.get("trusted_connection", "").lower() in {"yes", "true"} else (
                f"user={kv.get('uid', '?')}"
            )
            return (
                f"MSSQL {kv.get('server', '?')}/{kv.get('database', '?')} "
                f"({auth}, driver={kv.get('driver', '?')}, source=DB_CONN_STR)"
            )

        auth = "Windows auth" if self.mssql_trusted_connection else f"user={self.mssql_user}"
        return (
            f"MSSQL {self.mssql_server}:{self.mssql_port}/{self.mssql_database} "
            f"({auth}, driver={self.mssql_driver!r})"
        )

    @property
    def target_database(self) -> str:
        """Name of the database the pipeline writes to, whatever its source."""
        if self.db_conn_str:
            return self._parse_odbc(self.db_conn_str).get("database", "")
        return self.mssql_database

    def odbc_connection_string_for_database(self, database: str) -> str:
        """Return the active ODBC string re-pointed at another database.

        Used to reach ``master`` for server-level statements such as
        ``CREATE DATABASE``.
        """
        parts = []
        replaced = False
        for chunk in self.odbc_connection_string.split(";"):
            key = chunk.partition("=")[0].strip().lower()
            if key == "database":
                parts.append(f"Database={database}")
                replaced = True
            elif chunk.strip():
                parts.append(chunk)
        if not replaced:
            parts.append(f"Database={database}")
        return ";".join(parts)

    @staticmethod
    def _parse_odbc(conn_str: str) -> dict[str, str]:
        """Split an ODBC connection string into a lower-cased key -> value map.

        Braced values (``Driver={ODBC Driver 17 for SQL Server}``) are unwrapped;
        secrets are dropped rather than returned.
        """
        out: dict[str, str] = {}
        for chunk in conn_str.split(";"):
            if "=" not in chunk:
                continue
            key, _, value = chunk.partition("=")
            key = key.strip().lower()
            if key in {"pwd", "password"}:
                continue
            out[key] = value.strip().strip("{}")
        return out

    def require_youtube_api_key(self) -> str:
        """Return the YouTube Data API key or fail loudly."""
        if not self.youtube_api_key:
            raise ConfigurationError("YOUTUBE_API_KEY is not set; the collector service cannot run.")
        return self.youtube_api_key

    def require_openai_api_key(self) -> str:
        """Return the OpenAI API key or fail loudly."""
        if not self.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is not set; the AI analyzer service cannot run.")
        return self.openai_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide singleton :class:`Settings` instance."""
    return Settings()


# Convenience import target: `from core.config import settings`
settings: Settings = get_settings()

__all__ = ["BASE_DIR", "ENV_PATH", "ConfigurationError", "Settings", "get_settings", "settings"]
