"""
Reward cache for Boltz-2 co-folding results.
Stores SMILES -> reward mappings to avoid recomputing expensive Boltz-2 predictions.
"""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from loguru import logger


class BoltzRewardCache:
    """
    A persistent cache for storing and retrieving molecule rewards from Boltz-2 co-folding.
    
    This class provides thread-safe caching of SMILES -> reward mappings
    using SQLite with WAL mode for concurrent access.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the database, creating tables and ensuring integrity."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._is_valid_sqlite():
            raise ValueError(f"Corrupted or invalid DB at {self.db_path}. Please delete it and try again.")

        with self._connect() as (conn, cursor):
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    smiles TEXT NOT NULL,
                    reward REAL NOT NULL,
                    info TEXT,
                    UNIQUE(smiles)
                )
            """)

    def _is_valid_sqlite(self) -> bool:
        """Check if the database file is a valid SQLite database."""
        if not self.db_path.exists():
            return True
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA schema_version")
                conn.execute("PRAGMA integrity_check(1)")
                conn.execute("SELECT COUNT(*) FROM sqlite_master")
            return True
        except sqlite3.DatabaseError as e:
            logger.warning(f"Database validation failed: {repr(e)}")
            return False

    @contextmanager
    def _connect(self):
        """Robust DB connection context manager with retry and safe cleanup."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        max_attempts = 5
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            conn = None
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA busy_timeout=5000")
                cursor = conn.cursor()

                try:
                    yield conn, cursor
                    conn.commit()
                    return  # Successful completion
                except Exception:
                    # If an exception occurs during yield, rollback and re-raise
                    try:
                        conn.rollback()
                    except Exception:
                        pass  # Ignore rollback errors
                    raise
                finally:
                    # Always close the connection
                    try:
                        conn.close()
                    except Exception:
                        pass  # Ignore close errors

            except sqlite3.OperationalError as e:
                last_exception = e
                if "database is locked" in str(e).lower() or "disk i/o error" in str(e).lower():
                    logger.warning(f"Cache DB error (attempt {attempt}/{max_attempts}): {repr(e)}")
                    if attempt < max_attempts:
                        time.sleep(0.1 * attempt * (1 + 0.1 * attempt))
                        continue
                # If it's not a retryable error or we've exhausted attempts, re-raise
                raise
            except Exception:
                # For non-operational errors, don't retry
                raise

        # If we get here, we've exhausted all attempts
        if last_exception:
            raise last_exception
        else:
            raise sqlite3.OperationalError("Failed to connect to database after all attempts")

    def get_hits(self, smiles_list: list[str]) -> dict[str, tuple[float, str]]:
        """
        Retrieve cached rewards for a list of SMILES.

        Args:
            smiles_list: List of SMILES strings to look up

        Returns:
            Dictionary mapping SMILES -> (reward, info) for found entries
        """
        if not smiles_list:
            return {}

        if not all(isinstance(s, str) for s in smiles_list):
            raise ValueError("All items in smiles_list must be strings")

        placeholders = ",".join("?" for _ in smiles_list)
        query = f"SELECT smiles, reward, info FROM entries WHERE smiles IN ({placeholders})"

        with self._connect() as (conn, cursor):
            cursor.execute(query, smiles_list)
            results = cursor.fetchall()
            return {smiles: (reward, info) for smiles, reward, info in results}

    def get_db_size(self, fast: bool = False) -> int:
        """Get current number of entries in the database"""
        with self._connect() as (conn, cursor):
            cursor.execute("BEGIN IMMEDIATE")
            if fast:
                # sqlite_sequence only exists if there's an AUTOINCREMENT column
                # Since we don't use AUTOINCREMENT, check if table exists first
                try:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
                    if cursor.fetchone() is not None:
                        result = cursor.execute("SELECT seq FROM sqlite_sequence WHERE name='entries'").fetchone()
                        if result is not None:
                            return result[0]
                except sqlite3.OperationalError:
                    pass
                # Fall back to COUNT if sqlite_sequence doesn't exist or query fails
                return cursor.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            else:
                return cursor.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def insert_entries(self, entries: list[tuple[str, float, str]]):
        """
        Insert new entries into the cache.

        Args:
            entries: List of tuples (smiles, reward, info) to insert
        """
        if not entries:
            return

        # Validate entry format
        for i, entry in enumerate(entries):
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise ValueError(
                    f"Entry {i} must be a tuple of length 3, got {type(entry)} "
                    f"with length {len(entry) if hasattr(entry, '__len__') else 'unknown'}"
                )

            smiles, reward, info = entry
            if not isinstance(smiles, str):
                raise ValueError(f"Entry {i}: smiles must be string, got {type(smiles)}")
            if not isinstance(reward, (int, float)):
                raise ValueError(f"Entry {i}: reward must be numeric, got {type(reward)}")
            if not isinstance(info, str):
                raise ValueError(f"Entry {i}: info must be string, got {type(info)}")

        with self._connect() as (conn, cursor):
            cursor.execute("BEGIN IMMEDIATE")
            cursor.executemany(
                "INSERT OR REPLACE INTO entries (smiles, reward, info) VALUES (?, ?, ?)",
                entries,
            )

