#!/usr/bin/env python3
"""
KONO Schema Migration for GRIP
Adds 4D spatial memory grid and evolutionary instance management.

Part of GRIP-KONO implementation.
"""

import apsw
from pathlib import Path
from typing import Optional

from lib.hal.kono.constants import DB_PATH


def get_schema_version(conn: apsw.Connection) -> int:
    """Get current KONO schema version."""
    try:
        result = list(conn.execute(
            "SELECT value FROM kono_meta WHERE key = 'schema_version'"
        ))
        return int(result[0][0]) if result else 0
    except apsw.SQLError:
        return 0


def set_schema_version(conn: apsw.Connection, version: int):
    """Set KONO schema version."""
    conn.execute("""
        INSERT OR REPLACE INTO kono_meta (key, value)
        VALUES ('schema_version', ?)
    """, (str(version),))


# ---------------------------------------------------------------------------
# V1 migration helpers (decomposed for KISS compliance)
# ---------------------------------------------------------------------------

# Columns added to the embeddings table by KONO
_KONO_COLUMNS = [
    ("grid_x", "INTEGER DEFAULT 0"),
    ("grid_y", "INTEGER DEFAULT 0"),
    ("grid_z", "INTEGER DEFAULT 5"),
    ("dimension", "INTEGER DEFAULT 0"),
    ("fitness_score", "REAL DEFAULT 0.5"),
    ("last_accessed_at", "TEXT"),
    ("access_count", "INTEGER DEFAULT 0"),
    ("attestation_count", "INTEGER DEFAULT 0"),
    ("is_permanent", "INTEGER DEFAULT 0"),
    ("origin_instance_id", "TEXT"),
    ("source_platform", "TEXT"),
    ("source_channel_id", "TEXT"),
    ("source_sender_id", "TEXT"),
    ("source_interaction_id", "TEXT"),
]

_DIMENSION_RULES = [
    ("factual", 0, 2, "Facts, definitions, API references"),
    ("procedural", 3, 5, "How-to, workflows, processes"),
    ("episodic", 6, 8, "Session memories, user interactions"),
    ("emergent", 9, 11, "Patterns discovered, insights generated"),
    ("prompt", 0, 2, "User prompts (factual dimension)"),
    ("response", 3, 5, "AI responses (procedural dimension)"),
    ("concept", 6, 8, "Conceptual embeddings"),
    ("checkpoint", 6, 8, "Session checkpoints (episodic)"),
    ("plan", 3, 5, "Plans (procedural dimension)"),
]

_DECAY_PARAMS = [
    ("time_decay_rate", 0.95, "Fitness multiplied by this per day"),
    ("access_boost_rate", 0.1, "Fitness boost per access (capped)"),
    ("access_boost_cap", 2.0, "Maximum access boost multiplier"),
    ("permanence_threshold", 3, "Attestations needed for permanence"),
    ("fitness_validity_threshold", 0.8, "Minimum fitness for valid instance"),
]


def _migrate_columns(conn: apsw.Connection):
    """Add KONO columns to embeddings table (idempotent)."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(embeddings)")}
    for col_name, col_def in _KONO_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE embeddings ADD COLUMN {col_name} {col_def}")


def _migrate_tables(conn: apsw.Connection):
    """Create KONO core tables (idempotent)."""
    conn.execute("""CREATE TABLE IF NOT EXISTS kono_meta (
        key TEXT PRIMARY KEY, value TEXT,
        updated_at TEXT DEFAULT (datetime('now')))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS kono_domains (
        id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE NOT NULL,
        x_coord INTEGER NOT NULL, name TEXT,
        created_at TEXT DEFAULT (datetime('now')))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS kono_topics (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL,
        y_coord INTEGER NOT NULL, keywords TEXT,
        created_at TEXT DEFAULT (datetime('now')))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS kono_instances (
        id TEXT PRIMARY KEY, generation INTEGER NOT NULL, parent_id TEXT,
        skills_json TEXT, patterns_json TEXT, preferences_json TEXT,
        project_context TEXT, fitness REAL DEFAULT 0.5,
        created_at TEXT DEFAULT (datetime('now')), last_active_at TEXT,
        FOREIGN KEY (parent_id) REFERENCES kono_instances(id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS kono_attestations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        embedding_id INTEGER NOT NULL, instance_id TEXT NOT NULL,
        fitness_at_attestation REAL NOT NULL,
        attested_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (embedding_id) REFERENCES embeddings(id),
        FOREIGN KEY (instance_id) REFERENCES kono_instances(id),
        UNIQUE(embedding_id, instance_id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS kono_dimension_rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content_type TEXT NOT NULL, dimension_start INTEGER NOT NULL,
        dimension_end INTEGER NOT NULL, description TEXT,
        UNIQUE(content_type))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS kono_decay_config (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        param_name TEXT UNIQUE NOT NULL, param_value REAL NOT NULL,
        description TEXT)""")


def _migrate_seeds(conn: apsw.Connection):
    """Seed dimension rules and decay parameters."""
    for content_type, dim_start, dim_end, desc in _DIMENSION_RULES:
        conn.execute(
            "INSERT OR IGNORE INTO kono_dimension_rules "
            "(content_type, dimension_start, dimension_end, description) VALUES (?, ?, ?, ?)",
            (content_type, dim_start, dim_end, desc))
    for param_name, param_value, desc in _DECAY_PARAMS:
        conn.execute(
            "INSERT OR IGNORE INTO kono_decay_config "
            "(param_name, param_value, description) VALUES (?, ?, ?)",
            (param_name, param_value, desc))


def _migrate_indexes(conn: apsw.Connection):
    """Create performance indexes."""
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_embeddings_grid_x ON embeddings(grid_x)",
        "CREATE INDEX IF NOT EXISTS idx_embeddings_grid_y ON embeddings(grid_y)",
        "CREATE INDEX IF NOT EXISTS idx_embeddings_grid_z ON embeddings(grid_z)",
        "CREATE INDEX IF NOT EXISTS idx_embeddings_dimension ON embeddings(dimension)",
        "CREATE INDEX IF NOT EXISTS idx_embeddings_fitness ON embeddings(fitness_score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_embeddings_permanent ON embeddings(is_permanent) WHERE is_permanent = 1",
        "CREATE INDEX IF NOT EXISTS idx_embeddings_origin ON embeddings(origin_instance_id)",
        "CREATE INDEX IF NOT EXISTS idx_embeddings_platform ON embeddings(source_platform)",
        "CREATE INDEX IF NOT EXISTS idx_embeddings_channel ON embeddings(source_channel_id)",
        "CREATE INDEX IF NOT EXISTS idx_embeddings_sender ON embeddings(source_sender_id)",
        "CREATE INDEX IF NOT EXISTS idx_attestations_embedding ON kono_attestations(embedding_id)",
        "CREATE INDEX IF NOT EXISTS idx_attestations_instance ON kono_attestations(instance_id)",
        "CREATE INDEX IF NOT EXISTS idx_instances_generation ON kono_instances(generation)",
        "CREATE INDEX IF NOT EXISTS idx_instances_parent ON kono_instances(parent_id)",
        "CREATE INDEX IF NOT EXISTS idx_domains_path ON kono_domains(path)",
        "CREATE INDEX IF NOT EXISTS idx_topics_name ON kono_topics(name)",
    ]
    for sql in indexes:
        conn.execute(sql)


def _migrate_triggers(conn: apsw.Connection):
    """Create access tracking and attestation triggers."""
    conn.execute("""CREATE TRIGGER IF NOT EXISTS kono_track_access
        AFTER UPDATE OF last_accessed_at ON embeddings
        WHEN NEW.last_accessed_at IS NOT NULL
        BEGIN UPDATE embeddings
            SET access_count = access_count + 1,
                fitness_score = MIN(1.0, fitness_score * (1 + 0.1))
            WHERE id = NEW.id; END""")
    conn.execute("""CREATE TRIGGER IF NOT EXISTS kono_attestation_count_insert
        AFTER INSERT ON kono_attestations
        BEGIN UPDATE embeddings
            SET attestation_count = attestation_count + 1,
                is_permanent = CASE WHEN attestation_count + 1 >= 3 THEN 1
                    ELSE is_permanent END
            WHERE id = NEW.embedding_id; END""")


def migrate_v1(conn: apsw.Connection):
    """KONO v1 Migration: columns, tables, seeds, indexes, triggers."""
    _migrate_tables(conn)
    _migrate_columns(conn)
    _migrate_seeds(conn)
    _migrate_indexes(conn)
    _migrate_triggers(conn)


def init_kono_schema(conn: Optional[apsw.Connection] = None, db_path: Optional[Path] = None):
    """
    Initialize KONO schema with migrations.

    Args:
        conn: Existing connection (optional)
        db_path: Database path (optional, defaults to embeddings.db)
    """
    if conn is None:
        import sqlite_vec
        from lib.hal.kono.substrate import (
            is_lembed_healthy,
            _ensure_lembed_registered,
            LembedNotRegisteredError,
        )

        path = db_path or DB_PATH
        conn = apsw.Connection(str(path))
        conn.enableloadextension(True)
        conn.loadextension(sqlite_vec.loadable_path())
        if is_lembed_healthy():
            import sqlite_lembed
            conn.loadextension(sqlite_lembed.loadable_path())
            # Issue #199: register model on every new connection (see substrate.py)
            try:
                _ensure_lembed_registered(conn)
            except LembedNotRegisteredError:
                # Non-fatal here — schema migration doesn't need lembed.
                # Subsequent user queries that need lembed will surface the
                # typed exception via safe_lembed().
                pass

    current_version = get_schema_version(conn)

    if current_version < 1:
        migrate_v1(conn)
        set_schema_version(conn, 1)
        print("KONO schema v1 applied")

    return current_version


def _open_db() -> apsw.Connection:
    """Open embeddings DB with extensions loaded."""
    import sqlite_vec
    from lib.hal.kono.substrate import (
        is_lembed_healthy,
        _ensure_lembed_registered,
        LembedNotRegisteredError,
    )
    conn = apsw.Connection(str(DB_PATH))
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    if is_lembed_healthy():
        import sqlite_lembed
        conn.loadextension(sqlite_lembed.loadable_path())
        # Issue #199: ensure model registered to prevent NULL deref on lembed calls.
        try:
            _ensure_lembed_registered(conn)
        except LembedNotRegisteredError:
            pass
    return conn


def _cmd_init(conn: apsw.Connection) -> dict:
    prev = get_schema_version(conn)
    init_kono_schema(conn)
    return {"status": "ok", "previous_version": prev, "current_version": get_schema_version(conn)}


def _cmd_verify(conn: apsw.Connection) -> dict:
    expected_tables = ["kono_meta", "kono_domains", "kono_topics",
                       "kono_instances", "kono_attestations",
                       "kono_dimension_rules", "kono_decay_config"]
    existing = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'kono_%'")}
    missing = [t for t in expected_tables if t not in existing]
    emb_cols = {r[1] for r in conn.execute("PRAGMA table_info(embeddings)")}
    expected_cols = ["grid_x", "grid_y", "grid_z", "dimension",
                     "fitness_score", "is_permanent", "attestation_count",
                     "source_platform", "source_channel_id"]
    missing_cols = [c for c in expected_cols if c not in emb_cols]
    return {
        "status": "ok" if not missing and not missing_cols else "incomplete",
        "version": get_schema_version(conn),
        "tables": list(existing), "missing_tables": missing, "missing_columns": missing_cols}


def main():
    """CLI for KONO schema management."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="KONO Schema Manager")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("init", help="Initialize/upgrade KONO schema")
    sub.add_parser("version", help="Show schema version")
    sub.add_parser("verify", help="Verify schema integrity")
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    handlers = {
        "init": lambda c: _cmd_init(c),
        "version": lambda c: {"status": "ok", "version": get_schema_version(c)},
        "verify": lambda c: _cmd_verify(c),
    }

    conn = _open_db()
    print(json.dumps(handlers[args.command](conn), indent=2))
    conn.close()


if __name__ == "__main__":
    main()
