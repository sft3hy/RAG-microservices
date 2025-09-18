# utils/jira_connector.py

import requests
from requests.auth import HTTPBasicAuth
import logging
from typing import Dict, List, Optional
import streamlit as st

logger = logging.getLogger(__name__)


class JiraConnector:
    """Handles JIRA API connections and operations."""

    def __init__(self):
        self.base_url = None
        self.auth = None
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def authenticate(self, base_url: str, email: str, api_token: str) -> bool:
        """
        Authenticate with JIRA using email and API token.

        Args:
            base_url: JIRA instance URL (e.g., https://yourcompany.atlassian.net)
            email: User's email address
            api_token: API token generated from JIRA

        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            # Clean up base_url
            self.base_url = base_url.rstrip("/")
            self.auth = HTTPBasicAuth(email, api_token)

            # Test authentication by fetching current user info
            response = requests.get(
                f"{self.base_url}/rest/api/3/myself",
                auth=self.auth,
                headers=self.headers,
                timeout=10,
            )

            if response.status_code == 200:
                user_info = response.json()
                logger.info(
                    f"Successfully authenticated as {user_info.get('displayName')}"
                )
                return True
            else:
                logger.error(
                    f"Authentication failed: {response.status_code} - {response.text}"
                )
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during authentication: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during authentication: {str(e)}")
            return False

    def get_user_info(self) -> Optional[Dict]:
        """Get current user information."""
        if not self.auth:
            return None

        try:
            response = requests.get(
                f"{self.base_url}/rest/api/3/myself",
                auth=self.auth,
                headers=self.headers,
                timeout=10,
            )

            if response.status_code == 200:
                return response.json()
            return None

        except Exception as e:
            logger.error(f"Error fetching user info: {str(e)}")
            return None

    def get_projects(self) -> List[Dict]:
        """Get list of projects accessible to the user."""
        if not self.auth:
            return []

        try:
            response = requests.get(
                f"{self.base_url}/rest/api/3/project",
                auth=self.auth,
                headers=self.headers,
                timeout=10,
            )

            if response.status_code == 200:
                return response.json()
            return []

        except Exception as e:
            logger.error(f"Error fetching projects: {str(e)}")
            return []

    def search_issues(self, jql: str, max_results: int = 50) -> List[Dict]:
        """
        Search for issues using JQL.

        Args:
            jql: JIRA Query Language string
            max_results: Maximum number of results to return

        Returns:
            List of issue dictionaries
        """
        if not self.auth:
            return []

        try:
            params = {
                "jql": jql,
                "maxResults": max_results,
                "fields": "key,summary,description,status,priority,assignee,reporter,created,updated,issuetype,project",
            }

            response = requests.get(
                f"{self.base_url}/rest/api/3/search",
                auth=self.auth,
                headers=self.headers,
                params=params,
                timeout=15,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("issues", [])
            else:
                logger.error(f"Search failed: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            logger.error(f"Error searching issues: {str(e)}")
            return []

    def get_issue_details(self, issue_key: str) -> Optional[Dict]:
        """Get detailed information about a specific issue."""
        if not self.auth:
            return None

        try:
            response = requests.get(
                f"{self.base_url}/rest/api/3/issue/{issue_key}",
                auth=self.auth,
                headers=self.headers,
                timeout=10,
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get issue {issue_key}: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error fetching issue {issue_key}: {str(e)}")
            return None

    def get_issue_comments(self, issue_key: str) -> List[Dict]:
        """Get comments for a specific issue."""
        if not self.auth:
            return []

        try:
            response = requests.get(
                f"{self.base_url}/rest/api/3/issue/{issue_key}/comment",
                auth=self.auth,
                headers=self.headers,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("comments", [])
            return []

        except Exception as e:
            logger.error(f"Error fetching comments for {issue_key}: {str(e)}")
            return []


def format_issue_for_rag(issue: Dict, comments: List[Dict] = None) -> str:
    """
    Format a JIRA issue into a text format suitable for RAG processing.

    Args:
        issue: JIRA issue dictionary
        comments: Optional list of comments

    Returns:
        Formatted text string
    """
    fields = issue.get("fields", {})

    # Basic issue information
    formatted_text = f"""
JIRA Issue: {issue.get('key', 'Unknown')}
Summary: {fields.get('summary', 'No summary')}
Status: {fields.get('status', {}).get('name', 'Unknown')}
Priority: {fields.get('priority', {}).get('name', 'Unknown')}
Issue Type: {fields.get('issuetype', {}).get('name', 'Unknown')}
Project: {fields.get('project', {}).get('name', 'Unknown')}

Description:
{fields.get('description', 'No description provided')}

Reporter: {fields.get('reporter', {}).get('displayName', 'Unknown')}
Assignee: {fields.get('assignee', {}).get('displayName', 'Unassigned')}
Created: {fields.get('created', 'Unknown')}
Updated: {fields.get('updated', 'Unknown')}
"""

    # Add comments if provided
    if comments:
        formatted_text += "\n\nComments:\n"
        for i, comment in enumerate(comments, 1):
            author = comment.get("author", {}).get("displayName", "Unknown")
            created = comment.get("created", "Unknown")
            body = comment.get("body", "No content")

            formatted_text += f"\nComment {i} by {author} on {created}:\n{body}\n"

    return formatted_text.strip()


def get_jql_suggestions() -> List[str]:
    """Get common JQL query suggestions."""
    return [
        "assignee = currentUser() AND status != Done",
        "reporter = currentUser() ORDER BY created DESC",
        "project = YOUR_PROJECT AND status = 'In Progress'",
        "priority = High AND assignee = currentUser()",
        "created >= -7d ORDER BY created DESC",
        "updated >= -3d ORDER BY updated DESC",
        "status changed AFTER -1w",
        "text ~ 'bug' ORDER BY priority DESC",
    ]
