"""
RLM Database - SQLite backend for the External Memory Store (EMS).

Handles schema creation, chunk storage/retrieval, and indexing.
"""

from __future__ import annotations

import json
import sqlite3
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite

import logging

logger = logging.getLogger(__name__)


class RLMDatabase:
    """
    SQLite database manager for RLM memory chunks.
    
    Schema:
    - memory_chunks: Main table for compressed memory content
    - rcb_state: Current REPL Context Buffer entries
    - access_log: Usage tracking for forgetting curve
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize RLM database.
        
        Args:
            db_path: Path to SQLite database file. Defaults to ~/.lollmsbot/rlm_memory.db
        """
        if db_path is None:
            db_path = Path.home() / ".lollmsbot" / "rlm_memory.db"
        
        self.db_path: Path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
    
    async def initialize(self) -> None:
        """Initialize database connection and create schema if needed."""
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if we need to delete old/corrupted database
        await self._check_and_fix_schema()
        
        # Connect to database
        self._connection = await aiosqlite.connect(str(self.db_path))
        
        # Enable foreign keys
        await self._connection.execute("PRAGMA foreign_keys = ON")
        
        # Apply migrations first (before creating schema, to handle existing tables)
        await self._apply_migrations()
        
        # Create schema (for new databases or missing tables/indexes)
        await self._create_schema()
        
        await self._connection.commit()
        logger.info(f"RLM Database initialized at {self.db_path}")
    
    async def _check_and_fix_schema(self) -> None:
        """Check if database has correct schema, delete if not."""
        if not self.db_path.exists():
            return
        
        try:
            # Try to open and check schema
            conn = await aiosqlite.connect(str(self.db_path))
            
            # Check if memory_chunks table exists and has correct columns
            async with conn.execute("PRAGMA table_info(memory_chunks)") as cursor:
                columns = {row[1]: row for row in await cursor.fetchall()}
                
                # Check if the critical column exists
                has_content_compressed = "content_compressed" in columns
                has_content = "content" in columns
                
                if has_content and not has_content_compressed:
                    # Old schema detected - close and delete
                    await conn.close()
                    logger.warning("Old database schema detected (content column). Deleting and recreating...")
                    self.db_path.unlink()  # Delete the file
                    return
                elif not has_content_compressed:
                    # Missing critical column
                    await conn.close()
                    logger.warning("Corrupted database schema (missing content_compressed). Deleting and recreating...")
                    self.db_path.unlink()
                    return
            
            await conn.close()
            
        except Exception as e:
            # Database might be corrupted
            logger.warning(f"Database check failed: {e}. Will attempt to recreate...")
            try:
                if self.db_path.exists():
                    self.db_path.unlink()
            except:
                pass
    
    async def _create_schema(self) -> None:
        """Create database tables and indexes (safe for existing databases)."""
        # Main memory chunks table (CREATE IF NOT EXISTS - safe for existing)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS memory_chunks (
                chunk_id TEXT PRIMARY KEY,
                chunk_type TEXT NOT NULL,
                content_compressed BLOB NOT NULL,
                content_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                memory_importance REAL DEFAULT 1.0,
                compression_ratio REAL DEFAULT 1.0,
                tags TEXT,
                summary TEXT,
                load_hints TEXT,
                source TEXT,
                archived INTEGER DEFAULT 0
            )
        """)
        
        # RCB (REPL Context Buffer) state
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS rcb_state (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT,
                entry_type TEXT NOT NULL,
                content TEXT NOT NULL,
                display_order INTEGER DEFAULT 0,
                loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chunk_id) REFERENCES memory_chunks(chunk_id) ON DELETE SET NULL
            )
        """)
        
        # Access log for forgetting curve calculations
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS access_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_type TEXT,
                context_info TEXT,
                FOREIGN KEY (chunk_id) REFERENCES memory_chunks(chunk_id) ON DELETE CASCADE
            )
        """)
        
        # Self-knowledge cache (high-importance facts about the agent)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS self_knowledge (
                knowledge_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL DEFAULT 10.0,
                confirmed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confirmed_by TEXT
            )
        """)
        
        # Indexes that don't depend on potentially missing columns
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_type ON memory_chunks(chunk_type)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_importance ON memory_chunks(memory_importance DESC)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_access_log_chunk ON access_log(chunk_id, accessed_at)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_rcb_order ON rcb_state(display_order)
        """)
        
        # Note: idx_chunks_archived is handled in _apply_migrations to ensure column exists first
    
    async def _apply_migrations(self) -> None:
        """Apply schema migrations for existing databases."""
        # Define all columns that should exist in memory_chunks with their defaults
        required_columns = [
            ("content_compressed", "BLOB"),
            ("content_hash", "TEXT"),
            ("memory_importance", "REAL DEFAULT 1.0"),
            ("compression_ratio", "REAL DEFAULT 1.0"),
            ("tags", "TEXT"),
            ("summary", "TEXT"),
            ("load_hints", "TEXT"),
            ("source", "TEXT"),
            ("archived", "INTEGER DEFAULT 0"),
        ]
        
        for col_name, col_type in required_columns:
            try:
                # Try to query the column - if it fails, we need to add it
                async with self._connection.execute(
                    f"SELECT {col_name} FROM memory_chunks LIMIT 1"
                ) as cursor:
                    await cursor.fetchone()
                # Column exists
            except sqlite3.OperationalError as e:
                if "no such column" in str(e).lower():
                    try:
                        await self._connection.execute(f"""
                            ALTER TABLE memory_chunks ADD COLUMN {col_name} {col_type}
                        """)
                        logger.info(f"Added column {col_name} to memory_chunks")
                    except sqlite3.OperationalError:
                        # Table might not exist yet, ignore - _create_schema will handle it
                        pass
                else:
                    # Different error, table might not exist
                    pass
        
        # Create archived index (only if column now exists)
        try:
            await self._connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_archived ON memory_chunks(archived)
            """)
        except sqlite3.OperationalError:
            pass
    
    async def _migrate_content_to_content_compressed(self) -> None:
        """Migrate old table with 'content' column to new schema with 'content_compressed'."""
        try:
            # Rename old table
            await self._connection.execute("ALTER TABLE memory_chunks RENAME TO memory_chunks_old")
            
            # Create new table with correct schema
            await self._connection.execute("""
                CREATE TABLE memory_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    chunk_type TEXT NOT NULL,
                    content_compressed BLOB NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    memory_importance REAL DEFAULT 1.0,
                    compression_ratio REAL DEFAULT 1.0,
                    tags TEXT,
                    summary TEXT,
                    load_hints TEXT,
                    source TEXT,
                    archived INTEGER DEFAULT 0
                )
            """)
            
            # Migrate data - compress the old content
            async with self._connection.execute("SELECT * FROM memory_chunks_old") as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    # row structure: chunk_id, chunk_type, content, ...
                    chunk_id = row[0]
                    chunk_type = row[1]
                    old_content = row[2] if row[2] else ""
                    
                    # Compress the content
                    import hashlib
                    content_bytes = old_content.encode('utf-8') if isinstance(old_content, str) else old_content
                    if not content_bytes:
                        content_bytes = b"[empty]"
                    compressed = zlib.compress(content_bytes, level=6)
                    content_hash = hashlib.sha256(content_bytes).hexdigest()[:16]
                    
                    # Insert into new table
                    await self._connection.execute(
                        """
                        INSERT INTO memory_chunks (
                            chunk_id, chunk_type, content_compressed, content_hash,
                            created_at, last_accessed, access_count, memory_importance,
                            compression_ratio, tags, summary, load_hints, source, archived
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk_id, chunk_type, compressed, content_hash,
                            row[3] if len(row) > 3 else datetime.now().isoformat(),
                            row[4] if len(row) > 4 else datetime.now().isoformat(),
                            row[5] if len(row) > 5 else 0,
                            row[6] if len(row) > 6 else 1.0,
                            len(content_bytes) / len(compressed) if compressed else 1.0,
                            row[8] if len(row) > 8 else None,
                            row[9] if len(row) > 9 else None,
                            row[10] if len(row) > 10 else None,
                            row[11] if len(row) > 11 else None,
                            row[12] if len(row) > 12 else 0,
                        )
                    )
            
            # Drop old table
            await self._connection.execute("DROP TABLE memory_chunks_old")
            await self._connection.commit()
            logger.info(f"Successfully migrated {len(rows)} chunks to new schema")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            # Try to recover by restoring old table name
            try:
                await self._connection.execute("ALTER TABLE memory_chunks_old RENAME TO memory_chunks")
            except:
                pass
            raise
    
    async def store_chunk(
        self,
        chunk_id: str,
        chunk_type: str,
        content_compressed: bytes,
        content_hash: str,
        memory_importance: float = 1.0,
        compression_ratio: float = 1.0,
        tags: Optional[List[str]] = None,
        summary: Optional[str] = None,
        load_hints: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> bool:
        """Store a new memory chunk."""
        try:
            # Ensure connection is established
            if self._connection is None:
                logger.error("Database connection not initialized")
                return False
            
            # Validate required parameters
            if not content_compressed or len(content_compressed) == 0:
                logger.error(f"Cannot store chunk {chunk_id}: content_compressed is empty")
                # Store a placeholder to satisfy NOT NULL constraint
                content_compressed = b"[empty_content]"
            
            if not chunk_id:
                logger.error("Cannot store chunk: chunk_id is empty")
                return False
            
            # Log what we're about to store
            logger.debug(f"Storing chunk {chunk_id}: type={chunk_type}, compressed_size={len(content_compressed)}, hash={content_hash}")
            
            await self._connection.execute(
                """
                INSERT INTO memory_chunks (
                    chunk_id, chunk_type, content_compressed, content_hash,
                    memory_importance, compression_ratio, tags, summary, load_hints, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    chunk_type,
                    content_compressed,
                    content_hash,
                    memory_importance,
                    compression_ratio,
                    json.dumps(tags) if tags else None,
                    summary,
                    json.dumps(load_hints) if load_hints else None,
                    source,
                )
            )
            await self._connection.commit()
            logger.debug(f"Successfully stored chunk {chunk_id} of type {chunk_type}")
            return True
        except sqlite3.IntegrityError as e:
            # Duplicate key or constraint violation
            logger.error(f"Integrity error storing chunk {chunk_id}: {e}")
            # Check if it's the content constraint
            if "content" in str(e).lower():
                logger.error("The database appears to have old schema with 'content' column instead of 'content_compressed'")
                logger.error("Please delete the database file and restart: %s", self.db_path)
            return False
        except sqlite3.OperationalError as e:
            # Database operation error (table doesn't exist, etc.)
            logger.error(f"Operational error storing chunk {chunk_id}: {e}")
            # Try to reinitialize schema
            try:
                await self._create_schema()
                await self._connection.commit()
                logger.info("Recreated schema after operational error")
            except Exception as schema_e:
                logger.error(f"Failed to recreate schema: {schema_e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error storing chunk {chunk_id}: {e}")
            return False
    
    async def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a chunk by ID."""
        async with self._connection.execute(
            "SELECT * FROM memory_chunks WHERE chunk_id = ? AND (archived IS NULL OR archived = 0)",
            (chunk_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return self._row_to_dict(cursor, row)
            return None
    
    async def update_access(self, chunk_id: str, access_type: str = "read") -> None:
        """Update access statistics for a chunk."""
        now = datetime.now().isoformat()
        
        # Update chunk stats
        await self._connection.execute(
            """
            UPDATE memory_chunks 
            SET last_accessed = ?, access_count = access_count + 1
            WHERE chunk_id = ?
            """,
            (now, chunk_id)
        )
        
        # Log access
        await self._connection.execute(
            "INSERT INTO access_log (chunk_id, access_type, accessed_at) VALUES (?, ?, ?)",
            (chunk_id, access_type, now)
        )
        
        await self._connection.commit()
    
    async def search_chunks(
        self,
        query: Optional[str] = None,
        chunk_types: Optional[List[str]] = None,
        min_importance: Optional[float] = None,
        limit: int = 10,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for chunks matching criteria.
        
        Returns list of (chunk_dict, relevance_score) tuples.
        """
        # Build query
        conditions = ["(archived IS NULL OR archived = 0)"]
        params: List[Any] = []
        
        if chunk_types:
            placeholders = ','.join(['?' for _ in chunk_types])
            conditions.append(f"chunk_type IN ({placeholders})")
            params.extend(chunk_types)
        
        if min_importance is not None:
            conditions.append("memory_importance >= ?")
            params.append(min_importance)
        
        # Simple keyword search in summary and tags
        if query:
            conditions.append("(summary LIKE ? OR tags LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
        
        where_clause = " AND ".join(conditions)
        
        # Order by importance and recency
        sql = f"""
            SELECT * FROM memory_chunks
            WHERE {where_clause}
            ORDER BY memory_importance DESC, last_accessed DESC
            LIMIT ?
        """
        params.append(limit)
        
        results = []
        async with self._connection.execute(sql, params) as cursor:
            async for row in cursor:
                chunk_dict = self._row_to_dict(cursor, row)
                # Simple relevance scoring
                score = chunk_dict.get("memory_importance", 1.0)
                results.append((chunk_dict, score))
        
        return results
    
    async def get_chunks_by_importance(
        self,
        min_importance: float = 0.0,
        max_importance: float = 10.0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get chunks within an importance range."""
        async with self._connection.execute(
            """
            SELECT * FROM memory_chunks 
            WHERE memory_importance >= ? AND memory_importance <= ? AND (archived IS NULL OR archived = 0)
            ORDER BY memory_importance DESC
            LIMIT ?
            """,
            (min_importance, max_importance, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_dict(cursor, row) for row in rows]
    
    async def archive_low_importance_chunks(self, threshold: float = 0.5) -> int:
        """Archive chunks below importance threshold (forgetting curve)."""
        await self._connection.execute(
            "UPDATE memory_chunks SET archived = 1 WHERE memory_importance < ?",
            (threshold,)
        )
        await self._connection.commit()
        
        # Return count of archived chunks
        async with self._connection.execute(
            "SELECT changes()"
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0
    
    async def delete_chunk(self, chunk_id: str) -> bool:
        """Permanently delete a chunk."""
        try:
            await self._connection.execute(
                "DELETE FROM memory_chunks WHERE chunk_id = ?",
                (chunk_id,)
            )
            await self._connection.commit()
            return True
        except Exception:
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        
        # Count chunks
        async with self._connection.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE archived IS NULL OR archived = 0"
        ) as cursor:
            row = await cursor.fetchone()
            stats["active_chunks"] = row[0] if row else 0
        
        async with self._connection.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE archived = 1"
        ) as cursor:
            row = await cursor.fetchone()
            stats["archived_chunks"] = row[0] if row else 0
        
        # Average importance
        async with self._connection.execute(
            "SELECT AVG(memory_importance) FROM memory_chunks WHERE archived IS NULL OR archived = 0"
        ) as cursor:
            row = await cursor.fetchone()
            stats["avg_importance"] = row[0] if row and row[0] else 0.0
        
        # Total accesses
        async with self._connection.execute(
            "SELECT SUM(access_count) FROM memory_chunks"
        ) as cursor:
            row = await cursor.fetchone()
            stats["total_accesses"] = row[0] if row and row[0] else 0
        
        return stats
    
    async def store_self_knowledge(
        self,
        knowledge_id: str,
        category: str,
        content: str,
        importance: float = 10.0,
        confirmed_by: Optional[str] = None,
    ) -> bool:
        """Store high-importance self-knowledge."""
        try:
            await self._connection.execute(
                """
                INSERT OR REPLACE INTO self_knowledge 
                (knowledge_id, category, content, importance, confirmed_by, confirmed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    knowledge_id,
                    category,
                    content,
                    importance,
                    confirmed_by,
                    datetime.now().isoformat(),
                )
            )
            await self._connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error storing self-knowledge {knowledge_id}: {e}")
            return False
    
    async def get_self_knowledge(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve self-knowledge entries."""
        if category:
            async with self._connection.execute(
                "SELECT * FROM self_knowledge WHERE category = ? ORDER BY importance DESC",
                (category,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_dict(cursor, row) for row in rows]
        else:
            async with self._connection.execute(
                "SELECT * FROM self_knowledge ORDER BY importance DESC"
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_dict(cursor, row) for row in rows]
    
    async def clear_rcb(self) -> None:
        """Clear all RCB entries."""
        await self._connection.execute("DELETE FROM rcb_state")
        await self._connection.commit()
    
    async def store_rcb_entry(
        self,
        entry_type: str,
        content: str,
        chunk_id: Optional[str] = None,
        display_order: int = 0,
    ) -> int:
        """Store an RCB entry. Returns entry_id."""
        cursor = await self._connection.execute(
            """
            INSERT INTO rcb_state (entry_type, content, chunk_id, display_order)
            VALUES (?, ?, ?, ?)
            """,
            (entry_type, content, chunk_id, display_order)
        )
        await self._connection.commit()
        return cursor.lastrowid
    
    async def get_rcb_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get current RCB entries ordered by display_order."""
        async with self._connection.execute(
            """
            SELECT * FROM rcb_state 
            ORDER BY display_order ASC, loaded_at DESC
            LIMIT ?
            """,
            (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_dict(cursor, row) for row in rows]
    
    async def remove_rcb_entry(self, entry_id: int) -> bool:
        """Remove an RCB entry."""
        try:
            await self._connection.execute(
                "DELETE FROM rcb_state WHERE entry_id = ?",
                (entry_id,)
            )
            await self._connection.commit()
            return True
        except Exception:
            return False
    
    def _row_to_dict(
        self,
        cursor: aiosqlite.Cursor,
        row: sqlite3.Row,
    ) -> Dict[str, Any]:
        """Convert a database row to dictionary."""
        columns = [description[0] for description in cursor.description]
        result = {}
        for idx, col in enumerate(columns):
            value = row[idx]
            # Parse JSON fields
            if col in ("tags", "load_hints", "context_info") and value:
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            result[col] = value
        return result
    
    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
