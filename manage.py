"""Small admin entry point for the workspace.

Usage::

    python manage.py check      # verify the MSSQL connection
    python manage.py createdb   # create the target database if it is missing
    python manage.py initdb     # create any missing tables
    python manage.py schema     # list tables and their column counts
"""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import inspect

from core.config import settings
from core.database import get_engine, healthcheck, init_db


def _cmd_check() -> int:
    print(settings.masked_summary())
    ok = healthcheck()
    print("reachable:", ok)
    return 0 if ok else 1


def _cmd_createdb() -> int:
    """Create the configured database if it does not exist yet.

    ``CREATE DATABASE`` cannot run inside a transaction, so this connects to
    ``master`` with pyodbc directly in autocommit mode. Existing databases are
    left completely untouched.
    """
    import pyodbc

    target = settings.target_database
    if not target:
        print("Could not determine the target database name from .env.", file=sys.stderr)
        return 1

    master_conn_str = settings.odbc_connection_string_for_database("master")
    with pyodbc.connect(master_conn_str, autocommit=True) as conn:
        cursor = conn.cursor()
        exists = cursor.execute("SELECT DB_ID(?)", target).fetchval() is not None
        if exists:
            print(f"Database [{target}] already exists - nothing to do.")
            return 0
        # Identifier cannot be parameterised; bracket-quote it instead.
        cursor.execute(f"CREATE DATABASE [{target.replace(']', ']]')}]")
        print(f"Created database [{target}].")
    return 0


def _cmd_initdb() -> int:
    if not healthcheck():
        print("Cannot reach the database - check DB_CONN_STR in .env.", file=sys.stderr)
        return 1
    init_db()
    print("Schema ready.")
    return 0


def _cmd_schema() -> int:
    inspector = inspect(get_engine())
    for table in sorted(inspector.get_table_names()):
        columns = inspector.get_columns(table)
        print(f"{table:12} {len(columns):>3} columns")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "createdb", "initdb", "schema"])
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    commands = {
        "check": _cmd_check,
        "createdb": _cmd_createdb,
        "initdb": _cmd_initdb,
        "schema": _cmd_schema,
    }
    return commands[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
