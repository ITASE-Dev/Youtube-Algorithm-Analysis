"""Small admin entry point for the workspace.

Usage::

    python manage.py check      # verify the MSSQL connection
    python manage.py createdb   # create the target database if it is missing
    python manage.py initdb     # create any missing tables
    python manage.py schema     # list tables and their column counts
    python manage.py addcols    # show ALTER TABLE for new model columns
    python manage.py addcols --apply
"""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import inspect, text

from core.config import settings
from core.database import get_engine, healthcheck, init_db
from core.models import Base


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


def _cmd_addcols(apply: bool = False) -> int:
    """Add nullable columns that exist on the models but not yet in MSSQL.

    ``create_all`` only creates missing *tables*; it never alters an existing
    one. This covers the common additive case during development.

    Deliberately limited: it only ever emits ``ALTER TABLE ... ADD`` for
    nullable columns. Drops, renames, type changes and anything touching data
    are out of scope -- use Alembic for those.
    """
    engine = get_engine()
    inspector = inspect(engine)
    statements: list[str] = []

    for table in Base.metadata.sorted_tables:
        if table.name not in inspector.get_table_names():
            print(f"{table.name}: table missing entirely -- run `initdb` first.")
            continue

        existing = {c["name"].lower() for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name.lower() in existing:
                continue
            if not column.nullable:
                print(f"SKIP {table.name}.{column.name}: NOT NULL columns need a manual migration.")
                continue
            ddl_type = column.type.compile(dialect=engine.dialect)
            statements.append(f"ALTER TABLE [{table.name}] ADD [{column.name}] {ddl_type} NULL;")

    if not statements:
        print("Database schema already matches the models.")
        return 0

    for statement in statements:
        print(statement)

    if not apply:
        print(f"\n{len(statements)} column(s) missing. Re-run with --apply to execute.")
        return 0

    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
    print(f"\nApplied {len(statements)} column addition(s).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "createdb", "initdb", "schema", "addcols"])
    parser.add_argument(
        "--apply", action="store_true", help="addcols: execute the ALTER statements instead of printing them."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "addcols":
        return _cmd_addcols(apply=args.apply)

    commands = {
        "check": _cmd_check,
        "createdb": _cmd_createdb,
        "initdb": _cmd_initdb,
        "schema": _cmd_schema,
    }
    return commands[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
