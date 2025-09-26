import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from .models import DatabaseManager


class DocumentOperations:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def insert_document(
        self,
        document_name: str,
        user_id: str,
        document_text: str,
        file_size: int,
        file_type: str,
    ) -> int:
        """Insert a new document and return its ID."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO documents (document_name, user_id, document_text, file_size, file_type)
            VALUES (?, ?, ?, ?, ?)
        """,
            (document_name, user_id, document_text, file_size, file_type),
        )

        document_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return document_id

    def delete_document(
        self,
        document_id: str,
    ) -> int:
        """
        Deletes a document from the database and returns the number of deleted rows.
        """
        conn = None
        rows_deleted = 0
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            # Use %s for parameter placeholders in psycopg2, not ?
            # Also, pass parameters as a tuple in the second argument of execute()
            cursor.execute(
                """
                DELETE FROM documents WHERE document_id = ?;
                """,
                (document_id,),
            )

            # Get the number of deleted rows
            rows_deleted = cursor.rowcount

            conn.commit()
            cursor.close()

        except Exception as error:
            print(error)
            if conn is not None:
                conn.rollback()
        finally:
            if conn is not None:
                conn.close()

        return rows_deleted

    def get_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a user."""
        conn = self.db_manager.get_connection()
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT document_id, document_name, upload_timestamp, file_size, file_type, processed
            FROM documents WHERE user_id = ?
            ORDER BY upload_timestamp DESC
        """,
            (user_id,),
        )

        documents = []
        for row in cursor.fetchall():
            documents.append(
                {
                    "document_id": row[0],
                    "document_name": row[1],
                    "upload_timestamp": row[2],
                    "file_size": row[3],
                    "file_type": row[4],
                    "processed": row[5],
                }
            )

        conn.close()
        return documents

    def mark_document_processed(self, document_id: int):
        """Mark a document as processed."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE documents SET processed = TRUE WHERE document_id = ?",
            (document_id,),
        )

        conn.commit()
        conn.close()


class QueryOperations:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def delete_query(self, query_id: int):
        """Delete a query by ID."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_queries WHERE query_id = ?", (query_id,))

        conn.commit()
        conn.close()
        print(f"Deleted query with id {query_id}")

    def delete_user_queries(self, user_id):
        """Delete all queries by user id."""
        print("trying to delete")
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_queries WHERE user_id = ?", (user_id,))

        conn.commit()
        conn.close()
        print(f"Deleted user queries with id {user_id}")

    def insert_query(
        self,
        user_query: str,
        answer_text: str,
        answer_sources: List[Dict],
        user_id: str,
        processing_time: float,
        chunks_used: int,
        tokens_used: int = 0,
    ) -> int:
        """Insert a user query and its answer."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        # Convert sources to JSON string
        # print("ANSWER SOURCES", answer_sources)
        sources_json = json.dumps(
            answer_sources,
            default=lambda o: (o.isoformat() if isinstance(o, datetime) else str(o)),
        )

        cursor.execute(
            """
            INSERT INTO user_queries 
            (user_query, answer_text, answer_sources_used, user_id, processing_time, chunks_used, tokens_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                user_query,
                answer_text,
                sources_json,
                user_id,
                processing_time,
                chunks_used,
                tokens_used,
            ),
        )

        query_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return query_id

    def get_user_queries(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent queries for a user."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT query_id, user_query, answer_text, answer_sources_used, 
                   timestamp, processing_time, chunks_used, tokens_used
            FROM user_queries 
            WHERE user_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        """,
            (user_id, limit),
        )

        queries = []
        for row in cursor.fetchall():
            # Parse sources JSON
            sources = json.loads(row[3]) if row[3] else []

            queries.append(
                {
                    "query_id": row[0],
                    "user_query": row[1],
                    "answer_text": row[2],
                    "answer_sources_used": sources,
                    "timestamp": row[4],
                    "processing_time": row[5],
                    "chunks_used": row[6],
                    "tokens_used": row[7],
                }
            )

        conn.close()
        return queries

    def get_all_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all recent queries from all users."""
        conn = self.db_manager.get_connection()
        # Use row factory to easily convert to dictionary
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT query_id, user_query, answer_text, answer_sources_used,
                timestamp, processing_time, chunks_used, user_id, tokens_used
            FROM user_queries
            ORDER BY timestamp ASC
            LIMIT ?
        """,
            (limit,),
        )

        queries = []
        for row in cursor.fetchall():
            query_dict = dict(row)
            # Parse sources JSON
            if query_dict.get("answer_sources_used"):
                query_dict["answer_sources"] = json.loads(
                    query_dict["answer_sources_used"]
                )
            else:
                query_dict["answer_sources"] = []

            # For clarity, rename the key
            query_dict["content"] = query_dict.pop("answer_text")

            queries.append(query_dict)

        conn.close()
        return queries

    def get_todays_total_tokens(self, user_id: Optional[str] = None) -> int:
        """
        Get the total number of tokens used today.

        Args:
            user_id: If provided, get tokens for specific user. If None, get total for all users.

        Returns:
            Total tokens used today
        """
        return self.db_manager.get_todays_total_tokens(user_id)


class UserOperations:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def create_or_update_user(
        self, email: str, display_name: Optional[str] = None
    ) -> str:
        """
        Create a new user or update existing user's last login.

        Args:
            email: User's email address
            display_name: User's display name (optional)

        Returns:
            user_id: The user's ID (same as email in this implementation)
        """
        user_id = email  # Using email as user_id for simplicity

        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        # Check if user exists
        cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
        existing_user = cursor.fetchone()

        if existing_user:
            # Update last login
            cursor.execute(
                """
                UPDATE users 
                SET last_login = CURRENT_TIMESTAMP,
                    display_name = COALESCE(?, display_name),
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
                """,
                (display_name, user_id),
            )
        else:
            # Create new user
            cursor.execute(
                """
                INSERT INTO users (user_id, email, display_name, first_login, last_login)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (user_id, email, display_name),
            )

        conn.commit()
        conn.close()
        return user_id

    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user information.

        Args:
            user_id: User's ID

        Returns:
            Dictionary with user information or None if not found
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT user_id, email, display_name, first_login, last_login,
                   total_queries, total_documents, is_active, created_at, updated_at
            FROM users 
            WHERE user_id = ?
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "user_id": row[0],
                "email": row[1],
                "display_name": row[2],
                "first_login": row[3],
                "last_login": row[4],
                "total_queries": row[5],
                "total_documents": row[6],
                "is_active": row[7],
                "created_at": row[8],
                "updated_at": row[9],
            }
        return None

    def update_user_stats(
        self, user_id: str, increment_queries: int = 0, increment_documents: int = 0
    ):
        """
        Update user statistics.

        Args:
            user_id: User's ID
            increment_queries: Number to add to total_queries
            increment_documents: Number to add to total_documents
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE users 
            SET total_queries = total_queries + ?,
                total_documents = total_documents + ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
            """,
            (increment_queries, increment_documents, user_id),
        )

        conn.commit()
        conn.close()

    def get_user_activity_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get detailed activity statistics for a user.

        Args:
            user_id: User's ID

        Returns:
            Dictionary with activity statistics
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        # Get document count
        cursor.execute("SELECT COUNT(*) FROM documents WHERE user_id = ?", (user_id,))
        doc_count = cursor.fetchone()[0]

        # Get query count
        cursor.execute(
            "SELECT COUNT(*) FROM user_queries WHERE user_id = ?", (user_id,)
        )
        query_count = cursor.fetchone()[0]

        # Get today's token usage
        today_tokens = self.db_manager.get_todays_total_tokens()

        # Get total token usage
        cursor.execute(
            "SELECT COALESCE(SUM(tokens_used), 0) FROM user_queries WHERE user_id = ?",
            (user_id,),
        )
        total_tokens = cursor.fetchone()[0]

        # Get recent activity (last 7 days)
        cursor.execute(
            """
            SELECT DATE(timestamp, 'localtime') as query_date, COUNT(*) as query_count
            FROM user_queries 
            WHERE user_id = ? AND timestamp >= DATE('now', '-7 days', 'localtime')
            GROUP BY DATE(timestamp, 'localtime')
            ORDER BY query_date DESC
            """,
            (user_id,),
        )
        recent_activity = cursor.fetchall()

        conn.close()

        return {
            "document_count": doc_count,
            "total_queries": query_count,
            "today_tokens": today_tokens,
            "total_tokens": total_tokens,
            "recent_activity": recent_activity,
        }

    def get_all_users_summary(self) -> list:
        """
        Get a summary of all users for admin purposes.

        Returns:
            List of dictionaries with user summaries
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT u.user_id, u.email, u.display_name, u.first_login, u.last_login,
                   u.total_queries, u.total_documents, u.is_active,
                   COALESCE(SUM(uq.tokens_used), 0) as total_tokens
            FROM users u
            LEFT JOIN user_queries uq ON u.user_id = uq.user_id
            GROUP BY u.user_id, u.email, u.display_name, u.first_login, u.last_login,
                     u.total_queries, u.total_documents, u.is_active
            ORDER BY u.last_login DESC
            """
        )

        users = []
        for row in cursor.fetchall():
            users.append(
                {
                    "user_id": row[0],
                    "email": row[1],
                    "display_name": row[2],
                    "first_login": row[3],
                    "last_login": row[4],
                    "total_queries": row[5],
                    "total_documents": row[6],
                    "is_active": row[7],
                    "total_tokens": row[8],
                }
            )

        conn.close()
        return users


class GroupOperations:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def create_group(
        self,
        group_name: str,
        group_admin: str,
        description: Optional[str] = None,
        initial_members: Optional[List[str]] = None,
        initial_documents: Optional[List[int]] = None,
    ) -> int:
        """
        Create a new group.

        Args:
            group_name: Name of the group
            group_admin: Email of the group administrator
            description: Optional description of the group
            initial_members: List of user emails to add as initial members
            initial_documents: List of document IDs to add to the group

        Returns:
            group_id: The ID of the created group
        """
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            # Convert lists to JSON strings
            members_json = json.dumps(initial_members or [])
            documents_json = json.dumps(initial_documents or [])

            cursor.execute(
                """
                INSERT INTO groups (group_name, group_admin, group_members,
                                  group_documents, description)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    group_name,
                    group_admin,
                    members_json,
                    documents_json,
                    description,
                ),
            )

            group_id = cursor.lastrowid
            conn.commit()
            return group_id

    def get_user_groups(self, user_email: str) -> List[Dict[str, Any]]:
        """
        Get all groups where the user is either admin or member.

        Args:
            user_email: Email of the user

        Returns:
            List of group dictionaries
        """
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT group_id, group_name, group_admin, group_members,
                       group_documents, group_created_timestamp, updated_at,
                       is_active, description
                FROM groups
                WHERE is_active = 1 AND (
                    group_admin = ? OR
                    json_each.value = ?
                )
                LEFT JOIN json_each(groups.group_members)
                ORDER BY group_created_timestamp DESC
                """,
                (user_email, user_email),
            )

            groups = []
            for row in cursor.fetchall():
                group_dict = dict(row)
                # Parse JSON fields
                try:
                    group_dict["group_members"] = json.loads(
                        group_dict["group_members"] or "[]"
                    )
                    group_dict["group_documents"] = json.loads(
                        group_dict["group_documents"] or "[]"
                    )
                except (json.JSONDecodeError, TypeError):
                    group_dict["group_members"] = []
                    group_dict["group_documents"] = []

                groups.append(group_dict)

            return groups

    def get_group_by_id(self, group_id: int) -> Optional[Dict[str, Any]]:
        """
        Get group details by ID.

        Args:
            group_id: ID of the group

        Returns:
            Group dictionary or None if not found
        """
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT group_id, group_name, group_admin, group_members,
                       group_documents, group_created_timestamp, updated_at,
                       is_active, description
                FROM groups
                WHERE group_id = ? AND is_active = 1
                """,
                (group_id,),
            )

            row = cursor.fetchone()
            if row:
                group_dict = dict(row)
                # Parse JSON fields
                try:
                    group_dict["group_members"] = json.loads(
                        group_dict["group_members"] or "[]"
                    )
                    group_dict["group_documents"] = json.loads(
                        group_dict["group_documents"] or "[]"
                    )
                except (json.JSONDecodeError, TypeError):
                    group_dict["group_members"] = []
                    group_dict["group_documents"] = []

                return group_dict
            return None

    def add_member_to_group(
        self, group_id: int, user_email: str, admin_email: str
    ) -> bool:
        """
        Add a member to a group (only admin can do this).

        Args:
            group_id: ID of the group
            user_email: Email of user to add
            admin_email: Email of the requesting user (must be admin)

        Returns:
            True if successful, False otherwise
        """
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            # First check if requesting user is admin
            cursor.execute(
                "SELECT group_members FROM groups WHERE group_id = ? AND group_admin = ? AND is_active = 1",
                (group_id, admin_email),
            )

            row = cursor.fetchone()
            if not row:
                return False  # Not admin or group doesn't exist

            # Get current members
            try:
                current_members = json.loads(row["group_members"] or "[]")
            except (json.JSONDecodeError, TypeError):
                current_members = []

            # Add new member if not already in group
            if user_email not in current_members:
                current_members.append(user_email)

                cursor.execute(
                    """
                    UPDATE groups
                    SET group_members = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE group_id = ?
                    """,
                    (json.dumps(current_members), group_id),
                )
                conn.commit()

            return True

    def remove_member_from_group(
        self, group_id: int, user_email: str, admin_email: str
    ) -> bool:
        """
        Remove a member from a group (only admin can do this).

        Args:
            group_id: ID of the group
            user_email: Email of user to remove
            admin_email: Email of the requesting user (must be admin)

        Returns:
            True if successful, False otherwise
        """
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            # First check if requesting user is admin
            cursor.execute(
                "SELECT group_members FROM groups WHERE group_id = ? AND group_admin = ? AND is_active = 1",
                (group_id, admin_email),
            )

            row = cursor.fetchone()
            if not row:
                return False  # Not admin or group doesn't exist

            # Get current members
            try:
                current_members = json.loads(row["group_members"] or "[]")
            except (json.JSONDecodeError, TypeError):
                current_members = []

            # Remove member if in group
            if user_email in current_members:
                current_members.remove(user_email)

                cursor.execute(
                    """
                    UPDATE groups
                    SET group_members = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE group_id = ?
                    """,
                    (json.dumps(current_members), group_id),
                )
                conn.commit()

            return True

    def add_document_to_group(
        self, group_id: int, document_id: int, admin_email: str
    ) -> bool:
        """
        Add a document to a group (only admin can do this).

        Args:
            group_id: ID of the group
            document_id: ID of document to add
            admin_email: Email of the requesting user (must be admin)

        Returns:
            True if successful, False otherwise
        """
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            # First check if requesting user is admin
            cursor.execute(
                "SELECT group_documents FROM groups WHERE group_id = ? AND group_admin = ? AND is_active = 1",
                (group_id, admin_email),
            )

            row = cursor.fetchone()
            if not row:
                return False  # Not admin or group doesn't exist

            # Get current documents
            try:
                current_documents = json.loads(row["group_documents"] or "[]")
            except (json.JSONDecodeError, TypeError):
                current_documents = []

            # Add new document if not already in group
            if document_id not in current_documents:
                current_documents.append(document_id)

                cursor.execute(
                    """
                    UPDATE groups
                    SET group_documents = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE group_id = ?
                    """,
                    (json.dumps(current_documents), group_id),
                )
                conn.commit()

            return True

    def remove_document_from_group(
        self, group_id: int, document_id: int, admin_email: str
    ) -> bool:
        """
        Remove a document from a group (only admin can do this).

        Args:
            group_id: ID of the group
            document_id: ID of document to remove
            admin_email: Email of the requesting user (must be admin)

        Returns:
            True if successful, False otherwise
        """
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            # First check if requesting user is admin
            cursor.execute(
                "SELECT group_documents FROM groups WHERE group_id = ? AND group_admin = ? AND is_active = 1",
                (group_id, admin_email),
            )

            row = cursor.fetchone()
            if not row:
                return False  # Not admin or group doesn't exist

            # Get current documents
            try:
                current_documents = json.loads(row["group_documents"] or "[]")
            except (json.JSONDecodeError, TypeError):
                current_documents = []

            # Remove document if in group
            if document_id in current_documents:
                current_documents.remove(document_id)

                cursor.execute(
                    """
                    UPDATE groups
                    SET group_documents = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE group_id = ?
                    """,
                    (json.dumps(current_documents), group_id),
                )
                conn.commit()

            return True

    def delete_group(self, group_id: int, admin_email: str) -> bool:
        """
        Delete a group (soft delete - mark as inactive).

        Args:
            group_id: ID of the group
            admin_email: Email of the requesting user (must be admin)

        Returns:
            True if successful, False otherwise
        """
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE groups
                SET is_active = 0, updated_at = CURRENT_TIMESTAMP
                WHERE group_id = ? AND group_admin = ?
                """,
                (group_id, admin_email),
            )

            success = cursor.rowcount > 0
            conn.commit()
            return success

    def update_group(
        self,
        group_id: int,
        admin_email: str,
        group_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        """
        Update group details (only admin can do this).

        Args:
            group_id: ID of the group
            admin_email: Email of the requesting user (must be admin)
            group_name: New name for the group
            description: New description for the group

        Returns:
            True if successful, False otherwise
        """
        if not group_name and description is None:
            return False  # Nothing to update

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            update_parts = []
            params = []

            if group_name:
                update_parts.append("group_name = ?")
                params.append(group_name)

            if description is not None:
                update_parts.append("description = ?")
                params.append(description)

            update_parts.append("updated_at = CURRENT_TIMESTAMP")
            params.extend([group_id, admin_email])

            # Note the change in placeholder from %s to ?
            # and the reordering of params
            set_clause = ", ".join(update_parts)
            sql = f"""
                UPDATE groups
                SET {set_clause}
                WHERE group_id = ? AND group_admin = ? AND is_active = 1
            """

            # The group_id and admin_email are now at the end of the params list.
            # We need to adjust the parameter order for the WHERE clause.
            final_params = params[:-2]
            final_params.extend([params[-2], params[-1]])

            cursor.execute(sql, final_params)

            success = cursor.rowcount > 0
            conn.commit()
            return success

    def get_groups_for_document(self, document_id: int) -> List[Dict[str, Any]]:
        """
        Get all groups that have access to a specific document.

        Args:
            document_id: ID of the document

        Returns:
            List of group dictionaries
        """
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT g.group_id, g.group_name, g.group_admin, g.group_created_timestamp
                FROM groups g, json_each(g.group_documents)
                WHERE g.is_active = 1 AND json_each.value = ?
                ORDER BY g.group_name
                """,
                (document_id,),
            )

            groups = []
            for row in cursor.fetchall():
                group_dict = dict(row)
                groups.append(group_dict)

            return groups


class AdminOperations:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def nuke_database(self):
        """
        Deletes all data from documents, document_chunks, and user_queries tables.
        This will leave the tables intact but empty, ready for deployment.
        """
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        try:
            print("Attempting to delete all queries, documents, and chunks...")

            # The order of deletion is important due to foreign key constraints.
            # (document_chunks has a foreign key to documents)
            cursor.execute("DELETE FROM document_chunks;")
            cursor.execute("DELETE FROM documents;")
            cursor.execute("DELETE FROM user_queries;")

            # Reset the auto-increment counters for the primary keys so new
            # entries start from 1. This is good for a clean deployment state.
            print("Resetting table primary key sequences...")
            cursor.execute(
                "DELETE FROM sqlite_sequence WHERE name IN ('documents', 'document_chunks', 'user_queries');"
            )

            conn.commit()
            print("Database successfully nuked. All specified data has been deleted.")

        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        stats = {}

        # Count documents
        cursor.execute("SELECT COUNT(*) FROM documents")
        stats["total_documents"] = cursor.fetchone()[0]

        # Count chunks
        cursor.execute("SELECT COUNT(*) FROM document_chunks")
        stats["total_chunks"] = cursor.fetchone()[0]

        # Count queries
        cursor.execute("SELECT COUNT(*) FROM user_queries")
        stats["total_queries"] = cursor.fetchone()[0]

        # Total tokens used
        cursor.execute("SELECT COALESCE(SUM(tokens_used), 0) FROM user_queries")
        stats["total_tokens_used"] = cursor.fetchone()[0]

        # Tokens used today
        stats["tokens_used_today"] = self.db_manager.get_todays_total_tokens()

        conn.close()
        return stats
