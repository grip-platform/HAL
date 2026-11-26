#!/usr/bin/env python3
"""
KONO Memory Substrate for GRIP
Orchestrates 4D memory storage, dimension routing, and evolutionary instance management.

This is the main interface for KONO operations.

Part of GRIP-KONO implementation (Gen 310).
"""

import apsw
import json
import logging
import os
import struct
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

import sys as _sys

from lib.hal.kono.constants import (
    HAL_DIR as CLAUDE_DIR,
    DB_PATH,
    MODEL_PATH,
    LEMBED_CACHE_PATH as _LEMBED_CACHE,
    SESSION_COUNTER_FILE,
)
from lib.hal.kono.core import (
    InstanceDNA, GRIPInstance, Position4D, MemoryBlock,
    calculate_fitness, route_dimension, DIMENSION_ROUTES
)
from lib.hal.kono.counters import (
    SessionCounterStore, is_operational_content,
    OPERATIONAL_CONTENT_PATTERNS,
)
from lib.hal.kono.lembed_guard import is_likely_crash_input, LembedInputTooRisky
from lib.hal.kono.grip_embeddings import ensure_importable as _ensure_grip_embeddings

_lembed_healthy: Optional[bool] = None
_LEMBED_CACHE_TTL = 3600

# Name under which the default embedding model is registered in every
# connection's `temp.lembed_models` table. Keep stable — all call sites use it.
_LEMBED_MODEL_NAME = 'default'


class LembedNotRegisteredError(RuntimeError):
    """Raised when a lembed call would execute against a connection that has
    no registered model. The underlying C (`embed_single`) deref's the model
    pointer unconditionally, so the only safe guard is Python-side pre-flight.
    See issue #199 for the segfault class this prevents.
    """


def safe_lembed(conn: apsw.Connection, text: str,
                model_name: Optional[str] = None) -> List[float]:
    """Embed ``text`` via lembed, with NULL-deref prevention.

    Python wrapper around `SELECT lembed(?, ?)` that verifies the model is
    registered on ``conn`` before stepping the cursor. Raises
    :class:`LembedNotRegisteredError` with diagnostic context rather than
    letting the C segfault propagate as a SIGSEGV that kills the whole
    process. This is the public entry point other HAL code should use when
    it wants an embedding — it replaces raw `conn.execute("SELECT lembed(...)")`
    call sites.

    Args:
        conn: apsw connection with sqlite_lembed loaded.
        text: Input text to embed.
        model_name: Override registered model name (defaults to the standard).

    Returns:
        Embedding vector as a list of floats.

    Raises:
        LembedInputTooRisky: when ``text`` trips the W3-derived crash guard
            (low-entropy or long repeat-run inputs that SIGSEGV embed_single
            per upstream sqlite-lembed #7/#20). Callers should degrade to
            bag-of-words rather than retrying.
    """
    if is_likely_crash_input(text):
        raise LembedInputTooRisky(
            f"safe_lembed rejected input (len={len(text)}) — would likely "
            f"SIGSEGV sqlite-lembed embed_single"
        )
    _ensure_lembed_registered(conn)
    name = model_name or _LEMBED_MODEL_NAME
    try:
        rows = list(conn.execute(
            "SELECT lembed(?, ?)",
            (name, text),
        ))
    except apsw.SQLError as e:
        raise LembedNotRegisteredError(
            f"lembed({name!r}, ...) failed after registration check: {e}"
        )
    if not rows or rows[0][0] is None:
        raise LembedNotRegisteredError(
            f"lembed({name!r}, ...) returned NULL — model handle invalid"
        )
    return _decode_lembed_blob(rows[0][0])


def _decode_lembed_blob(blob: bytes) -> List[float]:
    """Decode the float32 BLOB returned by ``SELECT lembed(...)`` into a list.

    sqlite-lembed stores embeddings as packed little-endian float32; this
    helper unpacks them without pulling numpy into the substrate module.
    """
    float_count = len(blob) // 4
    return list(struct.unpack(f"{float_count}f", blob))


def _read_lembed_cache() -> Optional[bool]:
    """Read cached lembed health status. Returns None if stale/missing."""
    try:
        if _LEMBED_CACHE.exists():
            age = time.time() - _LEMBED_CACHE.stat().st_mtime
            if age < _LEMBED_CACHE_TTL:
                return _LEMBED_CACHE.read_text(encoding="utf-8").strip() == "ok"
    except OSError:
        pass
    return None


def _probe_lembed_subprocess() -> bool:
    """Test lembed in a subprocess — returns False if it segfaults.

    The probe actually calls lembed() (embed_single in C) with the real model,
    not just extension loading. Loading succeeds but embed_single crashes with
    SIGSEGV on NULL model pointer — the old probe missed this.
    """
    import subprocess
    model_path = str(MODEL_PATH)
    probe_script = (
        "import apsw, sqlite_vec, sqlite_lembed; "
        "c = apsw.Connection(':memory:'); "
        "c.enableloadextension(True); "
        "c.loadextension(sqlite_vec.loadable_path()); "
        "c.loadextension(sqlite_lembed.loadable_path()); "
        f"c.execute(\"INSERT INTO temp.lembed_models(name, model) "
        f"SELECT 'probe', lembed_model_from_file('{model_path}')\"); "
        "list(c.execute(\"SELECT lembed('probe', 'health check')\")); "
        "print('ok')"
    )
    try:
        result = subprocess.run(
            [_sys.executable, "-c", probe_script],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return False


def _write_lembed_cache(healthy: bool) -> None:
    """Persist lembed health status to disk."""
    try:
        _LEMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _LEMBED_CACHE.write_text("ok" if healthy else "broken", encoding="utf-8")
    except OSError:
        pass


def is_lembed_healthy() -> bool:
    """Check if sqlite-lembed is safe to load (no SIGSEGV on this Python)."""
    global _lembed_healthy
    if _lembed_healthy is not None:
        return _lembed_healthy

    cached = _read_lembed_cache()
    if cached is not None:
        _lembed_healthy = cached
        return cached

    healthy = _probe_lembed_subprocess()
    _lembed_healthy = healthy
    _write_lembed_cache(healthy)

    if not healthy:
        logger.warning("sqlite-lembed health check failed — semantic search disabled")
    return healthy


DEFAULT_LEMBED_MODEL = "hal-default"


def _ensure_lembed_registered(
    conn: apsw.Connection,
    name: str = DEFAULT_LEMBED_MODEL,
    model_path: Optional[Path] = None,
) -> bool:
    """Idempotently register a lembed model on ``conn``.

    sqlite-lembed stores registered models in ``temp.lembed_models`` which is
    connection-scoped. Calling ``SELECT lembed(name, text)`` with a name that
    was never INSERTed on this connection dereferences a NULL model pointer
    inside ``embed_single`` and segfaults the entire process (issue #199).
    Every HAL-owned connection must therefore register before any embedding
    is attempted; this function is safe to call repeatedly.
    """
    if not is_lembed_healthy():
        return False
    path = model_path or MODEL_PATH
    if not path.exists():
        logger.warning("lembed model file missing: %s", path)
        return False
    try:
        existing = list(conn.execute(
            "SELECT 1 FROM temp.lembed_models WHERE name = ?", (name,)
        ))
        if existing:
            return True
        conn.execute(
            "INSERT INTO temp.lembed_models(name, model) "
            "SELECT ?, lembed_model_from_file(?)",
            (name, str(path)),
        )
        return True
    except apsw.SQLError as exc:
        logger.warning("lembed registration failed for %r: %s", name, exc)
        return False


def hal_embed(
    conn: apsw.Connection,
    text: str,
    name: str = DEFAULT_LEMBED_MODEL,
) -> Optional[bytes]:
    """Defensive embedding helper — always registers before embedding.

    Returns ``None`` if lembed is unhealthy, the model could not be registered,
    the input trips the crash guard, or the query produced a NULL result.
    Never segfaults on happy-path NULL — that class of failure is what this
    wrapper exists to prevent. Pathological inputs (low-entropy, long repeat
    runs) are rejected pre-call via :func:`is_likely_crash_input`.
    """
    if is_likely_crash_input(text):
        logger.warning(
            "hal_embed(): rejecting pathological input (len=%d) — guard tripped",
            len(text),
        )
        return None
    if not _ensure_lembed_registered(conn, name=name):
        return None
    try:
        row = conn.execute("SELECT lembed(?, ?)", (name, text)).fetchone()
    except apsw.Error as exc:
        logger.warning("lembed() SQL error: %s", exc)
        return None
    if row is None or row[0] is None:
        return None
    return row[0]


class KonoLifecycleMixin:
    """
    Mixin: database connection and instance persistence.

    ISP boundary: connect, init_session, load_instance, save_instance, close.
    All methods rely only on self.conn, self.db_path, self.current_instance,
    self._counters, and self._permanence_events — attributes defined on
    MemorySubstrate.__init__.
    """

    def connect(self) -> apsw.Connection:
        """Get or create database connection."""
        if self.conn is not None:
            return self.conn

        import sqlite_vec

        self.conn = apsw.Connection(str(self.db_path))
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA busy_timeout=5000')
        self.conn.execute('PRAGMA synchronous=NORMAL')
        self.conn.enableloadextension(True)
        self.conn.loadextension(sqlite_vec.loadable_path())

        if is_lembed_healthy():
            import sqlite_lembed
            self.conn.loadextension(sqlite_lembed.loadable_path())
            # Issue #199: register the default model on every new connection.
            # lembed state is per-connection; without this, any SQL path that
            # hits `SELECT lembed(name, ...)` segfaults in embed_single's
            # NULL deref. Silent-fallback on registration failure — the
            # caller will hit LembedNotRegisteredError on first use, which
            # surfaces Python-side instead of killing the process.
            try:
                _ensure_lembed_registered(self.conn)
            except LembedNotRegisteredError as e:
                logger.warning(
                    "lembed model registration failed on connect(): %s — "
                    "embedding SQL will raise LembedNotRegisteredError "
                    "instead of segfault", e,
                )

        return self.conn

    def init_session(
        self,
        instance_id: str,
        generation: int,
        project_path: str,
        skills: Optional[List[str]] = None
    ) -> GRIPInstance:
        """
        Initialize session with instance tracking.

        Args:
            instance_id: Instance ID (from GRIP)
            generation: Generation number
            project_path: Current project
            skills: List of loaded skills

        Returns:
            GRIPInstance for this session
        """
        conn = self.connect()

        # Check if instance exists
        existing = list(conn.execute(
            "SELECT * FROM kono_instances WHERE id = ?",
            (instance_id,)
        ))

        if existing:
            self.current_instance = self._instance_from_row(existing[0])
        else:
            # Create new instance
            dna = InstanceDNA(
                skills=skills or [],
                project_context=project_path
            )

            self.current_instance = GRIPInstance(
                id=instance_id,
                generation=generation,
                dna=dna,
                fitness=0.5
            )

            conn.execute("""
                INSERT INTO kono_instances
                (id, generation, parent_id, skills_json, patterns_json,
                 preferences_json, project_context, fitness, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                instance_id,
                generation,
                None,
                json.dumps(dna.skills),
                json.dumps(dna.patterns),
                json.dumps(dna.preferences),
                project_path,
                0.5,
                datetime.now(timezone.utc).isoformat()
            ))

        # Update last active
        conn.execute("""
            UPDATE kono_instances
            SET last_active_at = ?
            WHERE id = ?
        """, (datetime.now(timezone.utc).isoformat(), instance_id))

        # Reset session counters for new session
        self._counters.reset()
        self._permanence_events = []

        return self.current_instance

    def load_instance(self, instance_id: str) -> Optional[GRIPInstance]:
        """
        Load instance from database.

        Args:
            instance_id: Instance ID to load

        Returns:
            GRIPInstance or None if not found
        """
        conn = self.connect()

        result = list(conn.execute(
            "SELECT * FROM kono_instances WHERE id = ?",
            (instance_id,)
        ))

        if not result:
            return None

        return self._instance_from_row(result[0])

    def save_instance(self, instance: GRIPInstance) -> None:
        """
        Save instance state to database.

        Args:
            instance: Instance to save
        """
        conn = self.connect()

        conn.execute("""
            INSERT OR REPLACE INTO kono_instances
            (id, generation, parent_id, skills_json, patterns_json,
             preferences_json, project_context, fitness, created_at, last_active_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            instance.id,
            instance.generation,
            instance.parent_id,
            json.dumps(instance.dna.skills),
            json.dumps(instance.dna.patterns),
            json.dumps(instance.dna.preferences),
            instance.dna.project_context,
            instance.fitness,
            instance.created_at.isoformat(),
            instance.last_active_at.isoformat() if instance.last_active_at else None
        ))

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


class KonoEvolutionMixin:
    """
    Mixin: attestation, consensus, metabolism, evolution, consolidation.

    ISP boundary: attest, attest_recalled, mine, consensus,
    run_metabolism, evolve, consolidate (7 methods).
    """

    def attest(self, embedding_id: int) -> bool:
        """
        Add current instance's attestation to an embedding.

        Args:
            embedding_id: Embedding to attest

        Returns:
            True if attestation added (not duplicate)
        """
        if not self.current_instance:
            return False

        conn = self.connect()

        # Check for duplicate
        existing = list(conn.execute("""
            SELECT id FROM kono_attestations
            WHERE embedding_id = ? AND instance_id = ?
        """, (embedding_id, self.current_instance.id)))

        if existing:
            return False

        # Add attestation
        conn.execute("""
            INSERT INTO kono_attestations
            (embedding_id, instance_id, fitness_at_attestation)
            VALUES (?, ?, ?)
        """, (embedding_id, self.current_instance.id, self.current_instance.fitness))

        self._counters.attested += 1
        return True

    def attest_recalled(self, embedding_id: int) -> bool:
        """
        Attest a recalled embedding, ensuring an instance exists.

        Bumps access_count and last_accessed_at to prevent fitness decay
        on useful memories. Ensures an active instance exists so full
        attestation is recorded (contributing toward permanence threshold).

        Args:
            embedding_id: Embedding to attest

        Returns:
            True if access was recorded
        """
        conn = self.connect()

        # Ensure instance so attestation actually happens
        self._ensure_instance()

        # Bump last_accessed_at (trigger kono_track_access handles access_count)
        conn.execute("""
            UPDATE embeddings
            SET last_accessed_at = ?
            WHERE id = ?
        """, (datetime.now(timezone.utc).isoformat(), embedding_id))

        # Full attestation (instance guaranteed by _ensure_instance)
        # Note: self.attest() already increments counters.attested
        if self.current_instance:
            self.attest(embedding_id)
        else:
            # No instance available (DB error), still count for session stats
            self._counters.attested += 1

        return True

    def mine(
        self,
        content: str,
        embed_type: str,
        embedding_id: int,
        project_path: Optional[str] = None,
        provenance: Optional[dict] = None,
    ) -> MemoryBlock:
        """
        Mine a memory block with current instance attestation.

        Args:
            content: Memory content
            embed_type: Embedding type
            embedding_id: ID of stored embedding
            project_path: Project context

        Returns:
            MemoryBlock with attestation
        """
        # Classify and route
        memory_type, dimension = self.classify_memory(content, embed_type)

        # Get coordinates
        x = self.get_or_create_domain(project_path or "")
        y = self.get_or_create_topic(embed_type)

        position = Position4D(
            x=float(x),
            y=float(y),
            z=5.0,  # Default mid-abstraction
            t=time.time()
        )

        # Store coordinates
        instance_id = self.current_instance.id if self.current_instance else None
        self.store(embedding_id, position, dimension, instance_id)

        # Write provenance if provided
        if provenance and self.engine and self.engine.conn:
            self.engine.conn.execute(
                "UPDATE embeddings SET source_platform=?, source_channel_id=?, "
                "source_sender_id=?, source_interaction_id=? WHERE id=?",
                (provenance.get('platform'), provenance.get('channel_id'),
                 provenance.get('sender_id'), provenance.get('interaction_id'),
                 embedding_id))

        # Create block
        block = MemoryBlock(
            id=f"block_{embedding_id}",
            position=position,
            chunks=[embedding_id],
            dimension=dimension
        )

        # Add attestation
        if self.current_instance:
            block.add_attestation(self.current_instance.id, self.current_instance.fitness)
            self.attest(embedding_id)

        return block

    def consensus(self, min_attestations: int = 3) -> List[int]:
        """
        Get embeddings that have achieved consensus permanence.

        Args:
            min_attestations: Minimum attestation count

        Returns:
            List of permanent embedding IDs
        """
        conn = self.connect()

        results = list(conn.execute("""
            SELECT id FROM embeddings
            WHERE attestation_count >= ?
              AND is_permanent = 1
        """, (min_attestations,)))

        return [r[0] for r in results]

    def run_metabolism(self) -> int:
        """
        Run fitness decay on all embeddings based on time since access.

        Returns:
            Number of embeddings updated
        """
        conn = self.connect()
        config = self._get_decay_config()

        decay_rate = config.get("time_decay_rate", 0.95)
        boost_rate = config.get("access_boost_rate", 0.1)
        boost_cap = config.get("access_boost_cap", 2.0)

        # Get embeddings that need decay
        now = datetime.now(timezone.utc).isoformat()

        # Update fitness for non-permanent embeddings
        conn.execute("""
            UPDATE embeddings
            SET fitness_score = MIN(1.0, MAX(0.0,
                fitness_score * ? * (1 + MIN(?, ? * access_count))
            ))
            WHERE is_permanent = 0
              AND (last_accessed_at IS NULL
                   OR julianday(?) - julianday(last_accessed_at) > 1)
        """, (decay_rate, boost_cap, boost_rate, now))

        # Return count of affected rows (approximation)
        result = list(conn.execute(
            "SELECT changes()"
        ))
        return result[0][0] if result else 0

    def evolve(self, generations: int = 1) -> List[GRIPInstance]:
        """
        Run evolutionary selection on instances.

        High-fitness instances reproduce, low-fitness decay.
        Single-pass O(n) — decay factor scales with generation count.

        Args:
            generations: Number of generations to simulate

        Returns:
            List of surviving instances
        """
        conn = self.connect()
        config = self._get_decay_config()

        validity_threshold = config.get("fitness_validity_threshold", 0.8)
        survivors = []

        # Single pass: apply decay scaled by generation count (O(n) not O(n*g))
        instances = list(conn.execute("""
            SELECT id, generation, parent_id, skills_json, patterns_json,
                   preferences_json, project_context, fitness,
                   created_at, last_active_at
            FROM kono_instances
            ORDER BY fitness DESC
        """))

        for row in instances:
            inst = self._instance_from_row(row)

            # Apply time decay (scaled by generations for multi-gen simulation)
            if row[9]:  # last_active_at
                last_active = datetime.fromisoformat(row[9])
                days = (datetime.now(timezone.utc) - last_active).total_seconds() / 86400
                inst.metabolize(days * generations)

            # Check validity
            if inst.fitness >= validity_threshold:
                survivors.append(inst)

            # Update in database
            conn.execute("""
                UPDATE kono_instances SET fitness = ? WHERE id = ?
            """, (inst.fitness, inst.id))

        return survivors

    def consolidate(self) -> int:
        """
        Mark embeddings as permanent based on attestation consensus.

        Returns:
            Number of newly permanent embeddings
        """
        conn = self.connect()
        config = self._get_decay_config()

        threshold = int(config.get("permanence_threshold", 3))

        # Update permanence for embeddings meeting threshold
        conn.execute("""
            UPDATE embeddings
            SET is_permanent = 1
            WHERE attestation_count >= ?
              AND is_permanent = 0
        """, (threshold,))

        result = list(conn.execute("SELECT changes()"))
        newly = result[0][0] if result else 0

        self._counters.promoted += newly
        if newly > 0:
            self._permanence_events.append({
                'count': newly,
                'timestamp': time.time()
            })

        return newly

class KonoPruningMixin:
    """
    Mixin: memory pruning and cleanup.

    ISP boundary: prune (1 method). Separated from KonoEvolutionMixin
    to keep each mixin at ≤7 public methods.
    """

    def prune(
        self,
        project_path: Optional[str] = None,
        below_fitness: Optional[float] = None,
        memory_id: Optional[int] = None,
        include_permanent: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Prune memories from the substrate.

        Deletes from: embeddings, kono_attestations, vec_embeddings (best-effort).

        Args:
            project_path: Filter to specific project domain
            below_fitness: Delete memories below this fitness threshold
            memory_id: Delete a specific memory by ID
            include_permanent: Allow deleting permanent memories (default: protected)
            dry_run: Show candidates without deleting

        Returns:
            Dict with count of affected memories and details
        """
        conn = self.connect()

        # Build WHERE clause
        conditions: List[str] = []
        params: List[Any] = []

        if memory_id is not None:
            conditions.append("id = ?")
            params.append(memory_id)
        else:
            if project_path is not None:
                grid_x = self._resolve_project_grid_x(project_path)
                if grid_x is None:
                    return {"pruned": 0, "error": f"No domain found for: {project_path}"}
                conditions.append("grid_x = ?")
                params.append(grid_x)

            if below_fitness is not None:
                conditions.append("fitness_score < ?")
                params.append(below_fitness)

        if not include_permanent:
            conditions.append("is_permanent = 0")

        if not conditions:
            return {"pruned": 0, "error": "No filter specified (would delete everything)"}

        where = " AND ".join(conditions)

        # Find candidates
        candidates = list(conn.execute(
            f"SELECT id, content, grid_x, fitness_score, is_permanent FROM embeddings WHERE {where}",
            tuple(params)
        ))

        if dry_run:
            return {
                "dry_run": True,
                "candidates": len(candidates),
                "memories": [
                    {
                        "id": c[0],
                        "content": c[1][:150] if c[1] else "",
                        "grid_x": c[2],
                        "fitness": c[3],
                        "is_permanent": bool(c[4]),
                    }
                    for c in candidates
                ],
            }

        # Delete from all tables
        ids = [c[0] for c in candidates]
        if not ids:
            return {"pruned": 0}

        placeholders = ",".join("?" * len(ids))

        # 1. vec_embeddings (virtual table, DELETE may not be supported)
        vec_deleted = 0
        try:
            conn.execute(
                f"DELETE FROM vec_embeddings WHERE rowid IN ({placeholders})",
                tuple(ids)
            )
            vec_deleted = list(conn.execute("SELECT changes()"))[0][0]
        except Exception:
            pass  # Orphaned vectors are harmless

        # 2. kono_attestations
        conn.execute(
            f"DELETE FROM kono_attestations WHERE embedding_id IN ({placeholders})",
            tuple(ids)
        )
        att_deleted = list(conn.execute("SELECT changes()"))[0][0]

        # 3. embeddings (main table)
        conn.execute(
            f"DELETE FROM embeddings WHERE {where}",
            tuple(params)
        )
        emb_deleted = list(conn.execute("SELECT changes()"))[0][0]

        return {
            "pruned": emb_deleted,
            "attestations_deleted": att_deleted,
            "vectors_deleted": vec_deleted,
            "ids": ids,
        }


class KonoQueryMixin:
    """
    Mixin: semantic recall, statistics, session summaries, permanence events.

    ISP boundary: recall, get_compact_stats, get_session_summary,
    format_session_summary, get_stats, get_permanence_events,
    store_session_summary.
    """

    def recall(
        self,
        query: str,
        limit: int = 10,
        dimension: Optional[int] = None,
        min_fitness: float = 0.0,
        expand_spatial: bool = True,
        project_path: Optional[str] = None,
        namespace: Optional[str] = None,
        include_merged: bool = True,
        platform: Optional[str] = None,
        channel_id: Optional[str] = None,
        sender_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic recall - hybrid vector+keyword search with spatial expansion.

        Combines EmbeddingEngine.search_hybrid() for RRF-fused results with
        KONO spatial retrieval for neighborhood expansion.

        Args:
            query: Natural language search query
            limit: Maximum results to return
            dimension: Filter to specific dimension (0-11), or None for all
            min_fitness: Minimum fitness threshold
            expand_spatial: Whether to expand via KONO spatial neighbors
            project_path: Filter to specific project domain (None = no filter)
            namespace: Memory namespace filter (default: GRIP_ORG:default)
            include_merged: Include memories from merged namespaces

        Returns:
            List of matching memories with scores, positions, and metadata
        """
        self._ensure_instance()

        accessible_ns = self._resolve_accessible_namespaces(namespace, include_merged)

        filter_grid_x = None
        if project_path is not None:
            filter_grid_x = self._resolve_project_grid_x(project_path)
            if filter_grid_x is None:
                return []

        hybrid_hits = self._execute_hybrid_search(query, dimension, limit)
        if not hybrid_hits:
            return []

        results, seen_ids = self._enrich_and_filter(
            hybrid_hits, accessible_ns, filter_grid_x, min_fitness, dimension,
            platform=platform, channel_id=channel_id, sender_id=sender_id,
        )

        if expand_spatial and results:
            results.extend(self._expand_from_results(
                results[:3], limit, min_fitness, seen_ids,
                filter_grid_x=filter_grid_x,
            ))

        return self._finalize_results(results, limit)

    def get_compact_stats(self) -> str:
        """Return compact single-line stats for hook output."""
        stats = self.get_stats()
        total = stats['embeddings']['total']
        permanent = stats['embeddings']['permanent']
        instances = stats['instances']['total']
        valid = stats['instances']['valid']
        domains = stats['domains']
        return (
            f"{total} memories ({permanent} permanent) | "
            f"{instances} instances ({valid} active) | "
            f"{domains} domains"
        )

    def get_session_summary(self) -> Dict[str, int]:
        """Return session action summary counts."""
        return {
            'stored': self._counters.stored,
            'recalled': self._counters.recalled,
            'attested': self._counters.attested,
            'newly_permanent': self._counters.promoted,
        }

    def format_session_summary(self) -> str:
        """Return formatted session summary string."""
        s = self.get_session_summary()
        parts = []
        if s['stored']:
            parts.append(f"+{s['stored']} stored")
        if s['recalled']:
            parts.append(f"{s['recalled']} recalled")
        if s['attested']:
            parts.append(f"{s['attested']} attested")
        if s['newly_permanent']:
            parts.append(f"{s['newly_permanent']} promoted to permanent")
        return ', '.join(parts) if parts else 'no activity'

    def get_stats(self) -> Dict[str, Any]:
        """Get KONO substrate statistics."""
        conn = self.connect()

        # Embedding stats
        total = list(conn.execute("SELECT COUNT(*) FROM embeddings"))[0][0]
        permanent = list(conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE is_permanent = 1"
        ))[0][0]
        by_dimension = dict(conn.execute(
            "SELECT dimension, COUNT(*) FROM embeddings GROUP BY dimension"
        ))

        # Instance stats
        instances = list(conn.execute("SELECT COUNT(*) FROM kono_instances"))[0][0]
        valid_instances = list(conn.execute(
            "SELECT COUNT(*) FROM kono_instances WHERE fitness >= 0.8"
        ))[0][0]

        # Domain/topic stats
        domains = list(conn.execute("SELECT COUNT(*) FROM kono_domains"))[0][0]
        topics = list(conn.execute("SELECT COUNT(*) FROM kono_topics"))[0][0]

        return {
            "embeddings": {
                "total": total,
                "permanent": permanent,
                "by_dimension": by_dimension
            },
            "instances": {
                "total": instances,
                "valid": valid_instances
            },
            "domains": domains,
            "topics": topics
        }

    def get_permanence_events(self) -> List[Dict[str, Any]]:
        """Return permanence events since last check, then clear."""
        events = list(self._permanence_events)
        self._permanence_events.clear()
        return events

    def store_session_summary(
        self,
        generation: int,
        instance_id: str,
        message_count: int,
        achievement: str,
        project_path: Optional[str] = None,
    ) -> Optional[int]:
        """
        Store a session summary as an episodic KONO memory.

        Convenience wrapper around store_session_memory() that formats
        session metadata into a structured episodic memory. Called from
        session-end hook and context-gate 80% handler.

        Args:
            generation: GRIP generation number
            instance_id: Instance ID (full or truncated)
            message_count: Number of messages in session
            achievement: Session achievement string
            project_path: Project context path

        Returns:
            Embedding ID if stored, None on failure
        """
        content = (
            f"Gen {generation} | {instance_id[:8]} | "
            f"{message_count} msgs | {achievement}"
        )
        return self.store_session_memory(
            content=content,
            memory_type="episodic",
            project_path=project_path,
        )


class MemorySubstrate(KonoLifecycleMixin, KonoEvolutionMixin, KonoPruningMixin, KonoQueryMixin):
    """
    KONO Memory Substrate - orchestrates memory storage and retrieval.

    Responsibilities:
        - Route memories to appropriate dimensions
        - Manage 4D spatial coordinates
        - Track instance fitness and reproduction
        - Handle consensus-based permanence
        - Run metabolism (fitness decay) and consolidation

    Inherits:
        KonoLifecycleMixin  -- connect, init_session, load_instance, save_instance, close
        KonoEvolutionMixin  -- attest, attest_recalled, mine, consensus,
                               run_metabolism, evolve, consolidate, prune
        KonoQueryMixin      -- recall, get_compact_stats, get_session_summary,
                               format_session_summary, get_stats, get_permanence_events,
                               store_session_summary
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.conn: Optional[apsw.Connection] = None
        self.current_instance: Optional[GRIPInstance] = None
        self._decay_config: Optional[Dict[str, float]] = None

        # Session-scoped counters (persisted via SessionCounterStore)
        self._counters = SessionCounterStore()
        self._permanence_events: List[Dict[str, Any]] = []

    def _ensure_instance(self) -> Optional[GRIPInstance]:
        """
        Ensure an active instance exists for attestation.

        If no current_instance is set, creates an ephemeral session instance.
        This is the critical link that was missing - without an active instance,
        attest() silently no-ops and no memories ever reach permanence.

        Returns:
            Active GRIPInstance, or None if DB unavailable
        """
        if self.current_instance:
            return self.current_instance

        try:
            instance_id = f"auto-{uuid.uuid4().hex[:12]}"
            return self.init_session(
                instance_id=instance_id,
                generation=0,
                project_path=str(Path.cwd()),
            )
        except Exception:
            return None

    def _get_decay_config(self) -> Dict[str, float]:
        """Load decay configuration from database."""
        if self._decay_config:
            return self._decay_config

        conn = self.connect()
        result = list(conn.execute(
            "SELECT param_name, param_value FROM kono_decay_config"
        ))
        self._decay_config = {row[0]: row[1] for row in result}
        return self._decay_config

    # =========================================================================
    # DIMENSION ROUTING
    # =========================================================================

    def route(self, memory_type: str) -> Tuple[int, int, int]:
        """
        Route content type to dimension range.

        Args:
            memory_type: Type of memory (factual, procedural, episodic, emergent)

        Returns:
            Tuple of (start_dim, mid_dim, end_dim)
        """
        dims = DIMENSION_ROUTES.get(memory_type, (0, 1, 2))
        return dims

    def classify_memory(
        self,
        text: str,
        embed_type: str
    ) -> Tuple[str, int]:
        """
        Classify memory type and determine dimension.

        Args:
            text: Content text
            embed_type: Embedding type from storage

        Returns:
            Tuple of (memory_type, dimension)
        """
        # Check database rules first
        conn = self.connect()
        result = list(conn.execute(
            """SELECT dimension_start, dimension_end
               FROM kono_dimension_rules
               WHERE content_type = ?""",
            (embed_type,)
        ))

        if result:
            dim_start, dim_end = result[0]
            # Use middle of range
            dimension = (dim_start + dim_end) // 2
            memory_type = self._dimension_to_type(dimension)
            return (memory_type, dimension)

        # Fallback to default routing
        dimension = route_dimension(embed_type)
        memory_type = self._dimension_to_type(dimension)
        return (memory_type, dimension)

    @staticmethod
    def _instance_from_row(row: tuple) -> GRIPInstance:
        """Construct a GRIPInstance from a kono_instances DB row.

        Row format: (id, generation, parent_id, skills_json, patterns_json,
                      preferences_json, project_context, fitness,
                      [created_at, last_active_at])
        """
        kwargs = dict(
            id=row[0],
            generation=row[1],
            parent_id=row[2],
            dna=InstanceDNA(
                skills=json.loads(row[3]) if row[3] else [],
                patterns=json.loads(row[4]) if row[4] else {},
                preferences=json.loads(row[5]) if row[5] else {},
                project_context=row[6] or "",
            ),
            fitness=row[7],
        )
        if len(row) > 8 and row[8]:
            kwargs["created_at"] = datetime.fromisoformat(row[8])
        if len(row) > 9 and row[9]:
            kwargs["last_active_at"] = datetime.fromisoformat(row[9])
        return GRIPInstance(**kwargs)

    def _dimension_to_type(self, dimension: int) -> str:
        """Map dimension number to memory type."""
        if dimension <= 2:
            return 'factual'
        elif dimension <= 5:
            return 'procedural'
        elif dimension <= 8:
            return 'episodic'
        else:
            return 'emergent'

    # =========================================================================
    # DOMAIN AND TOPIC MANAGEMENT
    # =========================================================================

    def get_or_create_domain(self, project_path: str) -> int:
        """
        Get or create x-coordinate for a project.

        Args:
            project_path: Project directory path

        Returns:
            x-coordinate for this domain
        """
        conn = self.connect()

        # Check existing
        result = list(conn.execute(
            "SELECT x_coord FROM kono_domains WHERE path = ?",
            (project_path,)
        ))

        if result:
            return result[0][0]

        # Create new - find max x_coord and increment
        max_x = list(conn.execute(
            "SELECT COALESCE(MAX(x_coord), -1) FROM kono_domains"
        ))[0][0]

        new_x = max_x + 1
        name = Path(project_path).name if project_path else "unknown"

        conn.execute(
            "INSERT INTO kono_domains (path, x_coord, name) VALUES (?, ?, ?)",
            (project_path, new_x, name)
        )

        return new_x

    def get_or_create_topic(self, topic_name: str, keywords: str = "") -> int:
        """
        Get or create y-coordinate for a topic.

        Args:
            topic_name: Topic/concept name
            keywords: Associated keywords

        Returns:
            y-coordinate for this topic
        """
        conn = self.connect()

        # Check existing
        result = list(conn.execute(
            "SELECT y_coord FROM kono_topics WHERE name = ?",
            (topic_name,)
        ))

        if result:
            return result[0][0]

        # Create new
        max_y = list(conn.execute(
            "SELECT COALESCE(MAX(y_coord), -1) FROM kono_topics"
        ))[0][0]

        new_y = max_y + 1

        conn.execute(
            "INSERT INTO kono_topics (name, y_coord, keywords) VALUES (?, ?, ?)",
            (topic_name, new_y, keywords)
        )

        return new_y

    # =========================================================================
    # MEMORY STORAGE
    # =========================================================================

    def store(
        self,
        embedding_id: int,
        position: Position4D,
        dimension: int,
        instance_id: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> None:
        """
        Update embedding with KONO spatial coordinates.

        Args:
            embedding_id: ID of existing embedding
            position: 4D spatial position
            dimension: Memory dimension
            instance_id: Origin instance ID
            namespace: Memory namespace for org isolation (default: GRIP_ORG:default).
                       Stored for future filtering when schema supports it.
        """
        # Resolve namespace — now persisted in DB column (Gen 432)
        ns = namespace or (
            os.environ.get("GRIP_ORG", "exampleorg") + ":default"
        )
        conn = self.connect()

        conn.execute("""
            UPDATE embeddings SET
                grid_x = ?,
                grid_y = ?,
                grid_z = ?,
                dimension = ?,
                origin_instance_id = ?,
                namespace = ?
            WHERE id = ?
        """, (
            int(position.x),
            int(position.y),
            int(position.z),
            dimension,
            instance_id or (self.current_instance.id if self.current_instance else None),
            ns,
            embedding_id
        ))

        self._counters.stored += 1

    def retrieve(
        self,
        query_position: Position4D,
        dimension: Optional[int] = None,
        radius: float = 10.0,
        limit: int = 10,
        min_fitness: float = 0.0,
        namespace: Optional[str] = None,
        include_merged: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories near a spatial position.

        Args:
            query_position: Center of search
            dimension: Filter by dimension (optional)
            radius: Search radius
            limit: Max results
            min_fitness: Minimum fitness threshold
            namespace: Memory namespace for org isolation (default: GRIP_ORG:default).
                       Reserved for future filtering when schema supports it.
            include_merged: Whether to include memories from merged namespaces
                           (requires active brain merge). Reserved for future use.

        Returns:
            List of matching embeddings with distance
        """
        # Resolve namespace — now with DB filtering (Gen 432)
        ns = namespace or (
            os.environ.get("GRIP_ORG", "exampleorg") + ":default"
        )
        conn = self.connect()

        # Collect accessible namespaces (own + merged)
        accessible_ns = [ns]
        if include_merged:
            try:
                from brain_merge import BrainMergeManager
                mgr = BrainMergeManager()
                accessible_ns.extend(mgr.get_active_merges(ns))
            except (ImportError, Exception):
                pass

        # Build query with optional dimension and namespace filters
        dim_filter = "AND dimension = ?" if dimension is not None else ""
        ns_placeholders = ",".join("?" for _ in accessible_ns)
        ns_filter = f"AND namespace IN ({ns_placeholders})"

        params = [
            query_position.x - radius, query_position.x + radius,
            query_position.y - radius, query_position.y + radius,
            min_fitness
        ]
        params.extend(accessible_ns)
        if dimension is not None:
            params.append(dimension)
        params.append(limit)

        results = list(conn.execute(f"""
            SELECT id, content, metadata, grid_x, grid_y, grid_z,
                   dimension, fitness_score, is_permanent,
                   created_at
            FROM embeddings
            WHERE grid_x BETWEEN ? AND ?
              AND grid_y BETWEEN ? AND ?
              AND fitness_score >= ?
              {ns_filter}
              {dim_filter}
            ORDER BY fitness_score DESC
            LIMIT ?
        """, tuple(params)))

        output = []
        for row in results:
            pos = Position4D(x=row[3], y=row[4], z=row[5], t=0)
            distance = query_position.distance(pos)

            if distance <= radius:
                output.append({
                    "id": row[0],
                    "content": row[1],
                    "metadata": json.loads(row[2]) if row[2] else None,
                    "position": pos.to_tuple(),
                    "dimension": row[6],
                    "fitness": row[7],
                    "is_permanent": bool(row[8]),
                    "created_at": row[9],
                    "distance": distance
                })

        output.sort(key=lambda x: x["distance"])
        result = output[:limit]
        self._counters.recalled += len(result)
        return result

    # =========================================================================
    # SESSION MEMORY STORAGE (Active KONO Usage)
    # =========================================================================

    def _is_duplicate_content(self, content: str, memory_type: str,
                              hours: int = 4, threshold: float = 0.7) -> bool:
        """Check if similar content was stored recently (word overlap)."""
        try:
            conn = self.connect()
            rows = conn.execute(
                """SELECT content FROM embeddings
                   WHERE type = ? AND created_at > datetime('now', ?)
                   ORDER BY created_at DESC LIMIT 20""",
                (memory_type, f'-{hours} hours'),
            ).fetchall()
            if not rows:
                return False
            new_words = set(content.lower().split())
            if not new_words:
                return False
            for (existing,) in rows:
                existing_words = set(existing.lower().split())
                if not existing_words:
                    continue
                overlap = len(new_words & existing_words)
                smaller = min(len(new_words), len(existing_words))
                if smaller > 0 and overlap / smaller >= threshold:
                    return True
            return False
        except Exception:
            return False

    def store_session_memory(
        self,
        content: str,
        memory_type: str = 'episodic',
        project_path: Optional[str] = None
    ) -> Optional[int]:
        """
        Store a memory via GRIP's embedding pipeline, which embeds through the
        local Ollama daemon (nomic-embed-text). Without GRIP installed there is
        no pipeline, and this returns None (HAL#925).

        Generates vector embedding, stores in embeddings table, then assigns
        KONO spatial coordinates for 4D retrieval. Ensures an active instance
        exists so the memory receives its first attestation.

        Includes deduplication: episodic entries with >70% word overlap to
        entries stored in the last 4 hours are skipped (#1036).

        Args:
            content: Text content to store
            memory_type: Type of memory (episodic, factual, procedural, emergent)
            project_path: Project context path

        Returns:
            Embedding ID if stored, None on failure
        """
        # Guard: skip operational/test content (#770)
        if is_operational_content(content):
            logger.debug("Skipping operational content: %s", content[:60])
            return None

        # Guard: deduplicate episodic entries (#1036)
        if memory_type == 'episodic' and self._is_duplicate_content(content, memory_type):
            logger.debug("Skipping duplicate episodic content: %s", content[:60])
            return None

        _ensure_grip_embeddings()
        try:
            from embeddings import EmbeddingEngine
            from embeddings.storage import store_embedding
        except ImportError:
            return None

        # Ensure instance exists so attestation works
        self._ensure_instance()

        engine = EmbeddingEngine()
        resolved_path = project_path or str(Path.cwd())

        # Store through embeddings pipeline (generates vector embedding)
        embedding_id = store_embedding(
            engine,
            text=content,
            embed_type=memory_type,
            project_path=resolved_path,
        )

        # Use mine() instead of raw store() - this handles spatial coords AND attestation
        self.mine(
            content=content,
            embed_type=memory_type,
            embedding_id=embedding_id,
            project_path=resolved_path,
        )

        self._counters.stored += 1
        return embedding_id

    # =========================================================================
    # SEMANTIC RECALL HELPERS
    # =========================================================================

    def _resolve_project_grid_x(self, project_path: str) -> Optional[int]:
        """
        Resolve a project path to its grid_x coordinate via kono_domains.

        Uses longest-prefix match to handle subdirectory paths.

        Args:
            project_path: Absolute project path

        Returns:
            grid_x coordinate or None if no matching domain
        """
        conn = self.connect()
        domains = list(conn.execute(
            "SELECT path, x_coord FROM kono_domains ORDER BY LENGTH(path) DESC"
        ))

        for domain_path, x_coord in domains:
            if project_path.startswith(domain_path):
                return x_coord

        return None

    def _resolve_accessible_namespaces(
        self, namespace: Optional[str], include_merged: bool
    ) -> List[str]:
        """Resolve the set of namespaces to search across.

        Includes current org namespace, legacy namespace (exampleorg),
        and any actively merged namespaces from brain_merge.
        """
        ns = namespace or (
            os.environ.get("GRIP_ORG", "exampleorg") + ":default"
        )
        accessible_ns = [ns]
        legacy_ns = "exampleorg:default"
        if legacy_ns != ns and legacy_ns not in accessible_ns:
            accessible_ns.append(legacy_ns)
        if include_merged:
            try:
                from brain_merge import BrainMergeManager
                mgr = BrainMergeManager()
                accessible_ns.extend(mgr.get_active_merges(ns))
            except (ImportError, Exception):
                pass
        return accessible_ns

    def _execute_hybrid_search(
        self, query: str, dimension: Optional[int], limit: int
    ) -> List[Dict[str, Any]]:
        """FTS5 keyword search against the embeddings_fts virtual table.

        HAL-vendored simplification: GRIP's substrate delegates to
        embeddings.EmbeddingEngine.search_hybrid() for RRF-fused vector +
        FTS retrieval. HAL doesn't vendor the embeddings package, so this
        uses FTS5 alone. Recall quality is lower than GRIP's hybrid, but
        the path is standalone. Re-vendor if HAL needs vector recall.
        """
        conn = self.connect()
        sanitized = self._sanitize_fts_query(query)
        if not sanitized:
            return []

        sql = (
            "SELECT e.id, e.content, e.type, e.fitness_score, "
            "e.is_permanent, e.dimension, e.grid_x, e.grid_y, e.grid_z, "
            "e.namespace, e.created_at, -fts.rank AS score "
            "FROM embeddings_fts fts "
            "JOIN embeddings e ON e.id = fts.rowid "
            "WHERE embeddings_fts MATCH ? "
        )
        params: List[Any] = [sanitized]
        if dimension is not None:
            sql += "AND e.dimension = ? "
            params.append(dimension)
        sql += "ORDER BY fts.rank LIMIT ?"
        params.append(limit * 2)

        try:
            rows = list(conn.execute(sql, params))
        except apsw.SQLError:
            return []

        return [
            {
                "id": r[0],
                "content": r[1],
                "type": r[2],
                "fitness_score": r[3],
                "is_permanent": bool(r[4]),
                "dimension": r[5],
                "grid_x": r[6],
                "grid_y": r[7],
                "grid_z": r[8],
                "namespace": r[9],
                "created_at": r[10],
                "score": r[11],
                "similarity": r[11],
                "source": "fts5",
            }
            for r in rows
        ]

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Strip FTS5 meta-characters that would cause parse errors."""
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in query)
        tokens = [t for t in cleaned.split() if t]
        return " ".join(tokens)

    def _enrich_and_filter(
        self,
        hybrid_hits: List[Dict[str, Any]],
        accessible_ns: List[str],
        filter_grid_x: Optional[int],
        min_fitness: float,
        dimension: Optional[int],
        platform: Optional[str] = None,
        channel_id: Optional[str] = None,
        sender_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], set]:
        """Enrich hybrid search hits with KONO metadata and apply filters.

        Returns (results, seen_ids) tuple.
        """
        conn = self.connect()
        results = []
        seen_ids = set()
        ns_placeholders = ",".join("?" for _ in accessible_ns)

        # Build provenance filter clause
        prov_clause = ""
        prov_params: list = []
        if platform:
            prov_clause += " AND source_platform = ?"
            prov_params.append(platform)
        if channel_id:
            prov_clause += " AND source_channel_id = ?"
            prov_params.append(channel_id)
        if sender_id:
            prov_clause += " AND source_sender_id = ?"
            prov_params.append(sender_id)

        for hit in hybrid_hits:
            emb_id = hit.get("id")
            if emb_id in seen_ids:
                continue

            row = list(conn.execute(
                f"""SELECT grid_x, grid_y, grid_z, dimension, fitness_score,
                          is_permanent, attestation_count, created_at
                   FROM embeddings WHERE id = ?
                   AND namespace IN ({ns_placeholders}){prov_clause}""",
                (emb_id, *accessible_ns, *prov_params)
            ))
            if not row:
                continue

            r = row[0]
            fitness = r[4] if r[4] is not None else 0.5

            if filter_grid_x is not None and r[0] != filter_grid_x:
                continue
            if fitness < min_fitness:
                continue
            if dimension is not None and r[3] is not None and r[3] != dimension:
                continue

            hit_content = hit.get("content", "")
            if is_operational_content(hit_content):
                continue

            seen_ids.add(emb_id)
            results.append({
                "id": emb_id,
                "content": hit_content,
                "type": hit.get("type", ""),
                "dimension": r[3],
                "dimension_name": self._dimension_to_type(r[3]) if r[3] is not None else "unknown",
                "position": (r[0], r[1], r[2]),
                "fitness": fitness,
                "is_permanent": bool(r[5]),
                "attestation_count": r[6] or 0,
                "created_at": r[7],
                "score": hit.get("recency_adjusted_score",
                                 hit.get("final_score",
                                          hit.get("rrf_score",
                                                   hit.get("similarity", 0.0)))),
                "source": "hybrid_search",
            })

        return results, seen_ids

    def _finalize_results(
        self, results: List[Dict[str, Any]], limit: int
    ) -> List[Dict[str, Any]]:
        """Deduplicate, sort, truncate, and attest recalled memories."""
        deduped = {}
        for mem in results:
            mid = mem.get("id")
            if mid not in deduped:
                deduped[mid] = mem
        results = list(deduped.values())

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        final = results[:limit]

        for mem in final:
            emb_id = mem.get("id")
            if emb_id is not None:
                self.attest_recalled(emb_id)

        self._counters.recalled += len(final)
        return final

    def _expand_from_results(
        self,
        seed_results: List[Dict[str, Any]],
        limit: int,
        min_fitness: float,
        seen_ids: set,
        filter_grid_x: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Expand search results via KONO spatial proximity.

        Two-phase: collect all nearby memories, then deduplicate via hash set.
        O(seeds + total_nearby) instead of nested iteration.
        """
        # Phase 1: collect all query positions from seeds
        queries = []
        for seed in seed_results:
            pos = seed.get("position")
            if not pos or (pos[0] is None and pos[1] is None):
                continue
            x, y = float(pos[0] or 0), float(pos[1] or 0)
            if x == 0 and y == 0:
                continue
            queries.append(Position4D(x=x, y=y, z=5.0, t=0))

        # Phase 2: retrieve all nearby memories (sequential DB queries, flat collection)
        all_nearby: List[Dict[str, Any]] = []
        effective_min = max(min_fitness, 0.3)
        per_query_limit = max(1, limit // (3 * max(len(queries), 1)))
        for qpos in queries:
            all_nearby.extend(self.retrieve(
                query_position=qpos, radius=5.0,
                limit=per_query_limit, min_fitness=effective_min,
            ))

        # Phase 3: deduplicate and filter in single pass (hash set lookup O(1))
        return self._dedupe_spatial(all_nearby, seen_ids, filter_grid_x)

    def _dedupe_spatial(
        self,
        memories: List[Dict[str, Any]],
        seen_ids: set,
        filter_grid_x: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Deduplicate spatial expansion results. O(n) with hash set."""
        expanded = []
        for mem in memories:
            mid = mem["id"]
            if mid in seen_ids:
                continue
            if filter_grid_x is not None:
                mem_pos = mem.get("position")
                if mem_pos and int(mem_pos[0]) != filter_grid_x:
                    continue
            seen_ids.add(mid)
            expanded.append({
                "id": mid,
                "content": mem.get("content", ""),
                "type": "spatial_expansion",
                "dimension": mem.get("dimension"),
                "dimension_name": self._dimension_to_type(mem["dimension"])
                    if mem.get("dimension") is not None else "unknown",
                "position": mem.get("position"),
                "fitness": mem.get("fitness", 0),
                "is_permanent": mem.get("is_permanent", False),
                "attestation_count": 0,
                "created_at": mem.get("created_at"),
                "score": 1.0 - (mem.get("distance", 10.0) / 10.0),
                "source": "kono_spatial",
            })
        return expanded


if __name__ == "__main__":
    raise SystemExit(
        "HAL's vendored kono.substrate has no CLI. Use GRIP's lib/kono_cli.py."
    )


