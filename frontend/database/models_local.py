import sqlite3
from typing import Optional, Dict, Any


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    def init_database(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Users table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                display_name TEXT,
                first_login DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_queries INTEGER DEFAULT 0,
                total_documents INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Documents table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                document_text TEXT NOT NULL,
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                file_size INTEGER,
                file_type TEXT,
                processed BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            )
        """
        )

        # groups table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS groups (
                group_id SERIAL PRIMARY KEY,
                group_name TEXT NOT NULL,
                group_admin TEXT NOT NULL,
                group_members TEXT, -- JSON array of user emails
                group_documents TEXT, -- JSON array of document IDs
                group_created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                description TEXT,
                FOREIGN KEY (group_admin) REFERENCES users (user_id) ON DELETE CASCADE
            )
        """
        )

        # User queries table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_queries (
                query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_query TEXT NOT NULL,
                answer_text TEXT NOT NULL,
                answer_sources_used TEXT, -- JSON string of source references
                user_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                processing_time REAL,
                chunks_used INTEGER,
                tokens_used INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            )
        """
        )

        # Add tokens_used column if it doesn't exist (for existing databases)
        try:
            cursor.execute(
                "ALTER TABLE user_queries ADD COLUMN tokens_used INTEGER DEFAULT 0"
            )
        except sqlite3.OperationalError:
            # Column already exists
            pass

        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_last_login ON users(last_login)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_queries_user_id ON user_queries(user_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON user_queries(timestamp)"
        )

        conn.commit()
        conn.close()

    def get_todays_total_tokens(self, user_id: Optional[str] = None) -> int:
        """
        Get the total number of tokens used today (local time).

        Args:
            user_id: If provided, get tokens for specific user. If None, get total for all users.

        Returns:
            Total tokens used today
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        if user_id:
            cursor.execute(
                """
                SELECT COALESCE(SUM(tokens_used), 0)
                FROM user_queries 
                WHERE user_id = ? 
                AND DATE(timestamp, 'localtime') = DATE('now', 'localtime')
                """,
                (user_id,),
            )
        else:
            cursor.execute(
                """
                SELECT COALESCE(SUM(tokens_used), 0)
                FROM user_queries 
                WHERE DATE(timestamp, 'localtime') = DATE('now', 'localtime')
                """
            )

        total_tokens = cursor.fetchone()[0]
        conn.close()

        return total_tokens
