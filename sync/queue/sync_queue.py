# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Sync Queue — SQLite-backed public chunk queue
#
# When a user taps "Share to Public Pool":
#   1. Chunk gets SHA-256 hash, node ID, Ed25519 signature
#   2. pool_visibility flips to "public"
#   3. Chunk is added to sync queue
#   4. Queue persists across app restarts
#
# Queue supports: add, get pending, mark synced, build manifest, revoke.

import json
import sqlite3
import threading
import time
from typing import Dict, List, Optional

from sync.protocol.pool_model import (
    KnowledgeChunk,
    ManifestEntry,
    SyncManifest,
)


class SyncQueue:
    """SQLite-backed queue for knowledge chunks shared to the public pool."""

    def __init__(self, db_path: str = "sync_queue.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_db()
        self._migrate_db()

    def _migrate_db(self):
        """Add columns that may be missing from older databases."""
        try:
            with self._lock:
                cursor = self.conn.execute("PRAGMA table_info(public_chunks)")
                columns = {row[1] for row in cursor.fetchall()}
                if "public_key" not in columns:
                    self.conn.execute(
                        "ALTER TABLE public_chunks ADD COLUMN public_key TEXT NOT NULL DEFAULT ''"
                    )
                    self.conn.commit()
        except Exception:
            pass

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS public_chunks (
                    content_hash TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    origin_node_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    signature TEXT NOT NULL,
                    shared_at REAL NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    synced INTEGER DEFAULT 0,
                    public_key TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS sync_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT NOT NULL,
                    peer_node_id TEXT NOT NULL,
                    synced_at REAL NOT NULL,
                    FOREIGN KEY (content_hash) REFERENCES public_chunks(content_hash)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_synced
                    ON public_chunks(synced);

                CREATE INDEX IF NOT EXISTS idx_sync_log_hash
                    ON sync_log(content_hash);
            """)
            self.conn.commit()

    def add_to_queue(self, chunk: KnowledgeChunk) -> bool:
        """
        Add a public chunk to the sync queue.
        Returns True if newly added, False if duplicate (already exists).
        The chunk must have pool_visibility="public".
        """
        if chunk.pool_visibility != "public":
            raise ValueError("Only public chunks can be added to the sync queue")

        try:
            with self._lock:
                self.conn.execute(
                    """INSERT INTO public_chunks
                       (content_hash, content, origin_node_id, timestamp, signature, shared_at, metadata, public_key)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chunk.content_hash,
                        chunk.content,
                        chunk.origin_node_id,
                        chunk.timestamp,
                        chunk.signature,
                        chunk.shared_at or time.time(),
                        json.dumps(chunk.metadata),
                        getattr(chunk, 'public_key', '') or '',
                    ),
                )
                self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Duplicate content_hash
            return False

    def get_pending(self, limit: int = 100) -> List[KnowledgeChunk]:
        """Get chunks that haven't been synced to any peer yet."""
        with self._lock:
            cursor = self.conn.execute(
                """SELECT content_hash, content, origin_node_id, timestamp,
                          signature, shared_at, metadata, public_key
                   FROM public_chunks
                   WHERE synced = 0
                   ORDER BY shared_at ASC
                   LIMIT ?""",
                (limit,),
            )
            return [self._row_to_chunk(row) for row in cursor.fetchall()]

    def mark_synced(self, content_hash: str, peer_node_id: str) -> None:
        """Record that a chunk was synced to a specific peer."""
        now = time.time()
        with self._lock:
            self.conn.execute(
                "INSERT INTO sync_log (content_hash, peer_node_id, synced_at) VALUES (?, ?, ?)",
                (content_hash, peer_node_id, now),
            )
            self.conn.execute(
                "UPDATE public_chunks SET synced = 1 WHERE content_hash = ?",
                (content_hash,),
            )
            self.conn.commit()

    def get_manifest(self, node_id: str) -> SyncManifest:
        """Build a SyncManifest from all public chunks in the queue."""
        with self._lock:
            cursor = self.conn.execute(
                """SELECT content_hash, timestamp, LENGTH(content)
                   FROM public_chunks
                   ORDER BY timestamp ASC"""
            )
            entries = [
                ManifestEntry(hash=row[0], timestamp=row[1], size=row[2])
                for row in cursor.fetchall()
            ]

            # Find last sync time
            cursor = self.conn.execute("SELECT MAX(synced_at) FROM sync_log")
            row = cursor.fetchone()
            last_sync = row[0] if row and row[0] else 0.0

        return SyncManifest(
            node_id=node_id,
            chunks=entries,
            last_sync=last_sync,
        )

    def get_chunk(self, content_hash: str) -> Optional[KnowledgeChunk]:
        """Retrieve a chunk by its content hash."""
        with self._lock:
            cursor = self.conn.execute(
                """SELECT content_hash, content, origin_node_id, timestamp,
                          signature, shared_at, metadata, public_key
                   FROM public_chunks
                   WHERE content_hash = ?""",
                (content_hash,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    def get_chunks_by_hashes(self, hashes: List[str]) -> List[KnowledgeChunk]:
        """Retrieve multiple chunks by their content hashes."""
        if not hashes:
            return []
        placeholders = ",".join("?" * len(hashes))
        with self._lock:
            cursor = self.conn.execute(
                f"""SELECT content_hash, content, origin_node_id, timestamp,
                           signature, shared_at, metadata, public_key
                    FROM public_chunks
                    WHERE content_hash IN ({placeholders})""",
                hashes,
            )
            return [self._row_to_chunk(row) for row in cursor.fetchall()]

    def get_all_chunks(self, limit: int = 500) -> List[KnowledgeChunk]:
        """Get all public chunks regardless of sync status, newest first."""
        with self._lock:
            cursor = self.conn.execute(
                """SELECT content_hash, content, origin_node_id, timestamp,
                          signature, shared_at, metadata, public_key
                   FROM public_chunks
                   ORDER BY shared_at DESC
                   LIMIT ?""",
                (limit,),
            )
            return [self._row_to_chunk(row) for row in cursor.fetchall()]

    def get_random_chunks(self, limit: int = 10) -> List[KnowledgeChunk]:
        """Get random public chunks for query generation.

        Uses rowid sampling for large tables (O(limit) instead of O(N) full-table
        ORDER BY RANDOM()). Falls back to ORDER BY RANDOM() for small tables
        where the overhead is negligible.
        """
        total = self.chunk_count()
        if total <= 100:
            # Small table — ORDER BY RANDOM() is fine
            with self._lock:
                cursor = self.conn.execute(
                    """SELECT content_hash, content, origin_node_id, timestamp,
                              signature, shared_at, metadata, public_key
                       FROM public_chunks
                       ORDER BY RANDOM()
                       LIMIT ?""",
                    (limit,),
                )
                return [self._row_to_chunk(row) for row in cursor.fetchall()]

        # Large table — sample random rowids
        import random
        with self._lock:
            cursor = self.conn.execute("SELECT MIN(rowid), MAX(rowid) FROM public_chunks")
            min_id, max_id = cursor.fetchone()
            if min_id is None:
                return []

            # Over-sample to account for gaps in rowid sequence
            sample_size = min(limit * 3, max_id - min_id + 1)
            candidates = random.sample(range(min_id, max_id + 1), sample_size)
            placeholders = ",".join("?" * len(candidates))
            cursor = self.conn.execute(
                f"""SELECT content_hash, content, origin_node_id, timestamp,
                           signature, shared_at, metadata, public_key
                    FROM public_chunks
                    WHERE rowid IN ({placeholders})
                    LIMIT ?""",
                candidates + [limit],
            )
            return [self._row_to_chunk(row) for row in cursor.fetchall()]

    def chunk_count(self) -> int:
        """Return total number of public chunks in the queue."""
        with self._lock:
            cursor = self.conn.execute("SELECT COUNT(*) FROM public_chunks")
            return cursor.fetchone()[0]

    def remove_from_queue(self, content_hash: str) -> bool:
        """
        Remove a chunk from the public queue (revoke sharing).
        Returns True if the chunk existed and was removed.
        """
        with self._lock:
            cursor = self.conn.execute(
                "DELETE FROM public_chunks WHERE content_hash = ?",
                (content_hash,),
            )
            self.conn.commit()
        return cursor.rowcount > 0

    def stats(self) -> Dict:
        """Get queue statistics."""
        with self._lock:
            cursor = self.conn.execute("SELECT COUNT(*) FROM public_chunks")
            total = cursor.fetchone()[0]

            cursor = self.conn.execute("SELECT COUNT(*) FROM public_chunks WHERE synced = 0")
            pending = cursor.fetchone()[0]

            cursor = self.conn.execute("SELECT COUNT(*) FROM public_chunks WHERE synced = 1")
            synced = cursor.fetchone()[0]

            cursor = self.conn.execute("SELECT COUNT(*) FROM sync_log")
            sync_events = cursor.fetchone()[0]

        return {
            "total": total,
            "pending": pending,
            "synced": synced,
            "sync_events": sync_events,
        }

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def _row_to_chunk(self, row) -> KnowledgeChunk:
        """Convert a database row to a KnowledgeChunk."""
        return KnowledgeChunk(
            content_hash=row[0],
            content=row[1],
            origin_node_id=row[2],
            timestamp=row[3],
            signature=row[4],
            pool_visibility="public",
            shared_at=row[5],
            metadata=json.loads(row[6]) if row[6] else {},
            public_key=row[7] if len(row) > 7 else "",
        )
