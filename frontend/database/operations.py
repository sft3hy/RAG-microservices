import psycopg
from psycopg.rows import dict_row
import json
from typing import List, Dict, Any, Optional
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
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO documents (document_name, user_id, document_text, file_size, file_type)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING document_id
                """,
                    (document_name, user_id, document_text, file_size, file_type),
                )

                document_id = cursor.fetchone()[0]
                conn.commit()
                return document_id

    def delete_document(
        self,
        document_id: str,
    ) -> int:
        """
        Deletes a document from the database and returns the number of deleted rows.
        Note: Chunks are now stored in Weaviate and should be deleted separately.
        """
        conn = None
        rows_deleted = 0
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                DELETE FROM documents WHERE document_id = %s;
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
        with self.db_manager.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT document_id, document_name, upload_timestamp, file_size, file_type, processed
                    FROM documents WHERE user_id = %s
                    ORDER BY upload_timestamp DESC
                """,
                    (user_id,),
                )

                documents = cursor.fetchall()
                return [dict(row) for row in documents]

    def mark_document_processed(self, document_id: int):
        """Mark a document as processed."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE documents SET processed = TRUE WHERE document_id = %s",
                    (document_id,),
                )
                conn.commit()


class QueryOperations:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def delete_query(self, query_id: int):
        """Delete a query by ID."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM user_queries WHERE query_id = %s", (query_id,)
                )
                conn.commit()
                print(f"Deleted query with id {query_id}")

    def delete_user_queries(self, user_id):
        """Delete all queries by user id."""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_queries WHERE user_id = %s", (user_id,))

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
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Convert sources to JSON string
                sources_json = json.dumps(answer_sources)

                cursor.execute(
                    """
                    INSERT INTO user_queries 
                    (user_query, answer_text, answer_sources_used, user_id, processing_time, chunks_used, tokens_used)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING query_id
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

                query_id = cursor.fetchone()[0]
                conn.commit()
                return query_id

    def get_user_queries(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent queries for a user."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT query_id, user_query, answer_text, answer_sources_used, 
                           timestamp, processing_time, chunks_used, tokens_used
                    FROM user_queries 
                    WHERE user_id = %s
                    ORDER BY timestamp ASC
                    LIMIT %s
                """,
                    (user_id, limit),
                )

                queries = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    # Parse sources JSON
                    if row_dict["answer_sources_used"]:
                        try:
                            sources = json.loads(row_dict["answer_sources_used"])
                        except (json.JSONDecodeError, TypeError):
                            sources = []
                    else:
                        sources = []
                    row_dict["answer_sources"] = sources
                    queries.append(row_dict)

                return queries

    def get_all_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all recent queries from all users."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT query_id, user_query, answer_text, answer_sources_used,
                        timestamp, processing_time, chunks_used, user_id, tokens_used
                    FROM user_queries
                    ORDER BY timestamp ASC
                    LIMIT %s
                """,
                    (limit,),
                )

                queries = []
                for row in cursor.fetchall():
                    query_dict = dict(row)
                    # Parse sources JSON
                    if query_dict.get("answer_sources_used"):
                        try:
                            query_dict["answer_sources"] = json.loads(
                                query_dict["answer_sources_used"]
                            )
                        except (json.JSONDecodeError, TypeError):
                            query_dict["answer_sources"] = []
                    else:
                        query_dict["answer_sources"] = []

                    # For clarity, rename the key
                    query_dict["content"] = query_dict.pop("answer_text")

                    queries.append(query_dict)

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

        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Try to insert new user, or update last_login if exists
                cursor.execute(
                    """
                    INSERT INTO users (user_id, email, display_name, first_login, last_login)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id) 
                    DO UPDATE SET 
                        last_login = CURRENT_TIMESTAMP,
                        display_name = COALESCE(EXCLUDED.display_name, users.display_name),
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (user_id, email, display_name),
                )
                conn.commit()

        return user_id

    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user information.

        Args:
            user_id: User's ID

        Returns:
            Dictionary with user information or None if not found
        """
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT user_id, email, display_name, first_login, last_login,
                           total_queries, total_documents, is_active, created_at, updated_at
                    FROM users 
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )
                row = cursor.fetchone()

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
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE users 
                    SET total_queries = total_queries + %s,
                        total_documents = total_documents + %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s
                    """,
                    (increment_queries, increment_documents, user_id),
                )
                conn.commit()

    def get_user_activity_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get detailed activity statistics for a user.

        Args:
            user_id: User's ID

        Returns:
            Dictionary with activity statistics
        """
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Get document count
                cursor.execute(
                    "SELECT COUNT(*) FROM documents WHERE user_id = %s", (user_id,)
                )
                doc_count = cursor.fetchone()[0]

                # Get query count
                cursor.execute(
                    "SELECT COUNT(*) FROM user_queries WHERE user_id = %s", (user_id,)
                )
                query_count = cursor.fetchone()[0]

                # Get today's token usage
                today_tokens = self.db_manager.get_todays_total_tokens(user_id)

                # Get total token usage
                cursor.execute(
                    "SELECT COALESCE(SUM(tokens_used), 0) FROM user_queries WHERE user_id = %s",
                    (user_id,),
                )
                total_tokens = cursor.fetchone()[0]

                # Get recent activity (last 7 days)
                cursor.execute(
                    """
                    SELECT DATE(timestamp) as query_date, COUNT(*) as query_count
                    FROM user_queries 
                    WHERE user_id = %s AND timestamp >= CURRENT_DATE - INTERVAL '7 days'
                    GROUP BY DATE(timestamp)
                    ORDER BY query_date DESC
                    """,
                    (user_id,),
                )
                recent_activity = cursor.fetchall()

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
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
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
            with conn.cursor() as cursor:
                # Convert lists to JSON
                members_json = json.dumps(initial_members or [])
                documents_json = json.dumps(initial_documents or [])

                cursor.execute(
                    """
                    INSERT INTO groups (group_name, group_admin, group_members, 
                                      group_documents, description)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING group_id
                    """,
                    (
                        group_name,
                        group_admin,
                        members_json,
                        documents_json,
                        description,
                    ),
                )

                group_id = cursor.fetchone()[0]
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
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT group_id, group_name, group_admin, group_members, 
                           group_documents, group_created_timestamp, updated_at, 
                           is_active, description
                    FROM groups 
                    WHERE is_active = TRUE AND (
                        group_admin = %s OR 
                        group_members::text LIKE %s
                    )
                    ORDER BY group_created_timestamp DESC
                    """,
                    (user_email, f'%"{user_email}"%'),
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
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT group_id, group_name, group_admin, group_members, 
                           group_documents, group_created_timestamp, updated_at, 
                           is_active, description
                    FROM groups 
                    WHERE group_id = %s AND is_active = TRUE
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
            with conn.cursor() as cursor:
                # First check if requesting user is admin
                cursor.execute(
                    "SELECT group_members FROM groups WHERE group_id = %s AND group_admin = %s AND is_active = TRUE",
                    (group_id, admin_email),
                )

                row = cursor.fetchone()
                if not row:
                    return False  # Not admin or group doesn't exist

                # Get current members
                try:
                    current_members = json.loads(row[0] or "[]")
                except (json.JSONDecodeError, TypeError):
                    current_members = []

                # Add new member if not already in group
                if user_email not in current_members:
                    current_members.append(user_email)

                    cursor.execute(
                        """
                        UPDATE groups 
                        SET group_members = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE group_id = %s
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
            with conn.cursor() as cursor:
                # First check if requesting user is admin
                cursor.execute(
                    "SELECT group_members FROM groups WHERE group_id = %s AND group_admin = %s AND is_active = TRUE",
                    (group_id, admin_email),
                )

                row = cursor.fetchone()
                if not row:
                    return False  # Not admin or group doesn't exist

                # Get current members
                try:
                    current_members = json.loads(row[0] or "[]")
                except (json.JSONDecodeError, TypeError):
                    current_members = []

                # Remove member if in group
                if user_email in current_members:
                    current_members.remove(user_email)

                    cursor.execute(
                        """
                        UPDATE groups 
                        SET group_members = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE group_id = %s
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
            with conn.cursor() as cursor:
                # First check if requesting user is admin
                cursor.execute(
                    "SELECT group_documents FROM groups WHERE group_id = %s AND group_admin = %s AND is_active = TRUE",
                    (group_id, admin_email),
                )

                row = cursor.fetchone()
                if not row:
                    return False  # Not admin or group doesn't exist

                # Get current documents
                try:
                    current_documents = json.loads(row[0] or "[]")
                except (json.JSONDecodeError, TypeError):
                    current_documents = []

                # Add new document if not already in group
                if document_id not in current_documents:
                    current_documents.append(document_id)

                    cursor.execute(
                        """
                        UPDATE groups 
                        SET group_documents = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE group_id = %s
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
            with conn.cursor() as cursor:
                # First check if requesting user is admin
                cursor.execute(
                    "SELECT group_documents FROM groups WHERE group_id = %s AND group_admin = %s AND is_active = TRUE",
                    (group_id, admin_email),
                )

                row = cursor.fetchone()
                if not row:
                    return False  # Not admin or group doesn't exist

                # Get current documents
                try:
                    current_documents = json.loads(row[0] or "[]")
                except (json.JSONDecodeError, TypeError):
                    current_documents = []

                # Remove document if in group
                if document_id in current_documents:
                    current_documents.remove(document_id)

                    cursor.execute(
                        """
                        UPDATE groups 
                        SET group_documents = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE group_id = %s
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
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE groups 
                    SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                    WHERE group_id = %s AND group_admin = %s
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
            with conn.cursor() as cursor:
                update_parts = []
                params = []

                if group_name:
                    update_parts.append("group_name = %s")
                    params.append(group_name)

                if description is not None:
                    update_parts.append("description = %s")
                    params.append(description)

                update_parts.append("updated_at = CURRENT_TIMESTAMP")
                params.extend([group_id, admin_email])

                cursor.execute(
                    f"""
                    UPDATE groups 
                    SET {', '.join(update_parts)}
                    WHERE group_id = %s AND group_admin = %s AND is_active = TRUE
                    """,
                    params,
                )

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
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    SELECT group_id, group_name, group_admin, group_created_timestamp
                    FROM groups 
                    WHERE is_active = TRUE AND group_documents::text LIKE %s
                    ORDER BY group_name
                    """,
                    (f"%{document_id}%",),
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
        Deletes all data from documents, user_queries, and groups tables.
        This will leave the tables intact but empty, ready for deployment.
        Note: Chunks are now stored in Weaviate and should be handled separately.
        """
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    print("Attempting to delete all queries, documents, and groups...")

                    # Delete data from all tables
                    cursor.execute("DELETE FROM documents;")
                    cursor.execute("DELETE FROM user_queries;")
                    cursor.execute("DELETE FROM groups;")

                    # Reset the auto-increment counters for the primary keys so new
                    # entries start from 1. This is good for a clean deployment state.
                    print("Resetting table primary key sequences...")
                    cursor.execute(
                        "ALTER SEQUENCE documents_document_id_seq RESTART WITH 1;"
                    )
                    cursor.execute(
                        "ALTER SEQUENCE user_queries_query_id_seq RESTART WITH 1;"
                    )
                    cursor.execute("ALTER SEQUENCE groups_group_id_seq RESTART WITH 1;")

                    conn.commit()
                    print(
                        "Database successfully nuked. All specified data has been deleted."
                    )

                except psycopg.Error as e:
                    print(f"An error occurred: {e}")
                    conn.rollback()
                    raise

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                stats = {}

                # Count documents
                cursor.execute("SELECT COUNT(*) FROM documents")
                stats["total_documents"] = cursor.fetchone()[0]

                # Count queries
                cursor.execute("SELECT COUNT(*) FROM user_queries")
                stats["total_queries"] = cursor.fetchone()[0]

                # Count groups
                cursor.execute("SELECT COUNT(*) FROM groups WHERE is_active = TRUE")
                stats["total_groups"] = cursor.fetchone()[0]

                # Total tokens used
                cursor.execute("SELECT COALESCE(SUM(tokens_used), 0) FROM user_queries")
                stats["total_tokens_used"] = cursor.fetchone()[0]

                # Tokens used today
                stats["tokens_used_today"] = self.db_manager.get_todays_total_tokens()

                # Note: Chunks are now in Weaviate
                stats["note"] = "Chunks are now stored in Weaviate vector database"

                return stats
