# utils/ui.py

import os
import streamlit as st
import logging
import traceback
from utils.file_handler import process_document_upload
from utils.streamlit_utils import float_to_percent
from utils.jira_connector import (
    JiraConnector,
    format_issue_for_rag,
)
from datetime import datetime
from config.settings import settings
import time
from database.weav_operations import WeaviateManager

weav_manager = WeaviateManager()

logger = logging.getLogger(__name__)


def get_file_type_emoji(file_type):
    """Get emoji for a given file type."""
    file_type_emojis = {
        ".pdf": "📄",
        ".docx": "📝",
        ".doc": "📝",
        ".xlsx": "📊",
        ".xls": "📊",
        ".csv": "📊",
        ".pptx": "📋",
        ".ppt": "📋",
        ".html": "🌐",
        ".htm": "🌐",
        ".py": "🐍",
        ".js": "⚡",
        ".json": "🔧",
        ".eml": "📧",
        ".txt": "📃",
        ".md": "📝",
        ".xml": "🔧",
        ".yaml": "🔧",
        ".yml": "🔧",
    }
    return file_type_emojis.get(file_type.lower(), "📎")


def format_file_size(size_bytes):
    """Format file size into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f} KB"
    else:
        return f"{size_bytes/(1024*1024):.1f} MB"


def display_sources(sources):
    """Displays retrieved sources in an expander."""
    if not sources:
        return
    with st.expander(f"📚 Sources Used ({len(sources)} Chunks)", expanded=False):
        for i, source in enumerate(sources, 1):
            try:
                with st.expander(f"Chunk {i}"):
                    header = source.get("contextual_header", "No Header")
                    text = source.get("chunk_text", "Content not available.")
                    display_text = text[:1000] + "..." if len(text) > 1000 else text
                    st.write(f"📄 {header}")
                    st.markdown(f"> {display_text}")
                    meta_parts = []
                    if score := source.get("relevance_score"):
                        meta_parts.append(
                            f"Similarity score: {float_to_percent(score)}"
                        )
                    if chunk_id := source.get("chunk_id"):
                        meta_parts.append(f"Chunk ID: `{chunk_id}`")
                    if meta_parts:
                        st.caption(" | ".join(meta_parts))
            except Exception as e:
                st.error(f"Error displaying source {i}: {str(e)}")


def display_jira_login():
    """Display JIRA login interface."""
    st.subheader("🔗 Connect to JIRA")
    # Auto-reconnect if credentials are stored but connection is lost
    if (
        not st.session_state.get("jira_authenticated", False)
        and st.session_state.get("jira_url")
        and st.session_state.get("jira_email")
        and st.session_state.get("jira_token")
    ):

        with st.spinner("Reconnecting to JIRA..."):
            connector = JiraConnector()
            if connector.authenticate(
                st.session_state.jira_url,
                st.session_state.jira_email,
                st.session_state.jira_token,
            ):
                st.session_state.jira_connector = connector
                st.session_state.jira_authenticated = True
                st.session_state.jira_user_info = connector.get_user_info()
                st.success("🔄 Auto-reconnected to JIRA!")
                st.rerun()
    with st.expander(
        "JIRA Connection",
        expanded=not st.session_state.get("jira_authenticated", False),
    ):
        if not st.session_state.get("jira_authenticated", False):
            st.info(
                """
            **How to connect:**
            1. Enter your JIRA instance URL (e.g., https://yourcompany.atlassian.net)
            2. Use your email address
            3. Create an API token from JIRA Account Settings > Security > API Tokens
            """
            )

            with st.form("jira_login"):
                jira_url = st.text_input(
                    "JIRA Instance URL",
                    placeholder="https://yourcompany.atlassian.net",
                    help="Your organization's JIRA URL",
                )
                jira_email = st.text_input(
                    "Email Address", placeholder="your.email@company.com"
                )
                jira_token = st.text_input(
                    "API Token",
                    type="password",
                    help="Generate this from JIRA Account Settings > Security > API Tokens",
                )

                submitted = st.form_submit_button("Connect to JIRA", type="primary")

                if submitted and jira_url and jira_email and jira_token:
                    with st.spinner("Connecting to JIRA..."):
                        connector = JiraConnector()
                        if connector.authenticate(jira_url, jira_email, jira_token):
                            # Store connection info in session state
                            st.session_state.jira_connector = connector
                            st.session_state.jira_authenticated = True
                            st.session_state.jira_user_info = connector.get_user_info()
                            st.success("✅ Successfully connected to JIRA!")
                            # Store JIRA credentials for session persistence
                            st.session_state.jira_url = jira_url
                            st.session_state.jira_email = jira_email
                            st.session_state.jira_token = jira_token
                            st.rerun()
                        else:
                            st.error(
                                "❌ Failed to connect to JIRA. Please check your credentials."
                            )
        else:
            # Show connected status
            user_info = st.session_state.get("jira_user_info", {})
            st.success(
                f"✅ Connected as: {user_info.get('displayName', 'Unknown User')}"
            )
            st.caption(f"Email: {user_info.get('emailAddress', 'Unknown')}")

            if st.button("Disconnect", type="secondary"):
                # Clear JIRA session state
                for key in [
                    "jira_connector",
                    "jira_authenticated",
                    "jira_user_info",
                    "jira_projects",
                    "jira_issues",
                    "jira_selected_issues",
                    "jira_url",
                    "jira_email",
                    "jira_token",
                ]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()


def display_sidebar_jira_interface(components, settings):
    """Display JIRA ticket management interface in sidebar."""
    connector = st.session_state.jira_connector

    st.header("🎫 JIRA Tickets")

    # Quick stats
    issues = st.session_state.get("jira_issues", [])
    selected_issues = st.session_state.get("jira_selected_issues", {})

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Loaded", len(issues))
    with col2:
        st.metric("Selected", len(selected_issues))

    # Search Interface (Compact)
    with st.expander("🔍 Search Issues", expanded=False):
        # Quick search buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("My Open Issues", use_container_width=True):
                jql = (
                    "assignee = currentUser() AND status != Done ORDER BY updated DESC"
                )
                with st.spinner("Loading..."):
                    new_issues = connector.search_issues(jql, 20)
                    st.session_state.jira_issues = new_issues
                    st.rerun()

        with col2:
            if st.button("Recent Issues", use_container_width=True):
                jql = (
                    "assignee = currentUser() AND updated >= -7d ORDER BY updated DESC"
                )
                with st.spinner("Loading..."):
                    new_issues = connector.search_issues(jql, 20)
                    st.session_state.jira_issues = new_issues
                    st.rerun()

        # Custom JQL
        jql_query = st.text_area(
            "Custom JQL Query",
            placeholder="e.g., project = MYPROJECT AND status = 'In Progress'",
            height=80,
        )

        if st.button("Search", use_container_width=True) and jql_query:
            with st.spinner("Searching..."):
                new_issues = connector.search_issues(jql_query, 30)
                st.session_state.jira_issues = new_issues
                if new_issues:
                    st.success(f"Found {len(new_issues)} issues")
                else:
                    st.warning("No issues found")
                st.rerun()

    # Issue Selection (Compact)
    if issues:
        with st.expander(f"📝 Select Issues ({len(issues)} loaded)", expanded=True):
            # Clear selections button
            if selected_issues:
                if st.button("Clear All Selections", use_container_width=True):
                    st.session_state.jira_selected_issues = {}
                    st.rerun()

            # Issue list with checkboxes
            for issue in issues[:15]:  # Limit to 15 for sidebar space
                key = issue["key"]
                fields = issue["fields"]
                summary = fields.get("summary", "No summary")
                status = fields.get("status", {}).get("name", "Unknown")
                priority = fields.get("priority", {}).get("name", "Unknown")

                # Truncate summary for sidebar
                display_summary = summary[:40] + "..." if len(summary) > 40 else summary

                is_selected = st.checkbox(
                    f"**{key}**",
                    value=key in selected_issues,
                    key=f"sidebar_issue_{key}",
                )

                if is_selected and key not in selected_issues:
                    st.session_state.jira_selected_issues[key] = issue
                elif not is_selected and key in selected_issues:
                    del st.session_state.jira_selected_issues[key]

                st.caption(f"{display_summary}")
                st.caption(f"Status: {status} | Priority: {priority}")
                st.divider()

            if len(issues) > 15:
                st.caption(f"... and {len(issues) - 15} more issues")

    # Analysis Options (Compact)
    if selected_issues:
        with st.expander(
            f"🤖 AI Analysis ({len(selected_issues)} selected)", expanded=False
        ):
            analysis_type = st.selectbox(
                "Analysis Type",
                [
                    "Generate Solutions",
                    "Create Documentation",
                    "Find Similar Issues",
                    "Extract Requirements",
                    "Risk Assessment",
                ],
                key="sidebar_analysis_type",
            )

            if st.button(
                "Analyze Selected Issues", type="primary", use_container_width=True
            ):
                # Create analysis prompt and add to chat
                analysis_prompt = create_analysis_prompt(selected_issues, analysis_type)

                # Add message to chat
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": f"Analyze selected JIRA issues: {analysis_type}",
                    }
                )

                # Add to chat input for processing
                st.session_state.pending_jira_analysis = {
                    "prompt": analysis_prompt,
                    "type": analysis_type,
                    "issue_count": len(selected_issues),
                }

                st.success("Analysis queued! Check the chat area.")
                st.rerun()


def analyze_jira_issues(selected_issues, analysis_type, components, settings):
    """Analyze selected JIRA issues using RAG."""
    connector = st.session_state.jira_connector

    progress_bar = st.progress(0)
    status_text = st.empty()

    analysis_results = []

    for i, (issue_key, issue) in enumerate(selected_issues.items()):
        status_text.text(f"Analyzing {issue_key}...")

        try:
            # Get detailed issue information including comments
            detailed_issue = connector.get_issue_details(issue_key)
            comments = connector.get_issue_comments(issue_key)

            # Format issue for RAG processing
            formatted_issue = format_issue_for_rag(detailed_issue, comments)

            # Create analysis prompt based on type
            prompts = {
                "Generate Solution Suggestions": f"""
                Based on the following JIRA issue and relevant documentation, provide detailed solution suggestions:
                
                {formatted_issue}
                
                Please provide:
                1. Root cause analysis
                2. Step-by-step solution approach
                3. Potential risks and mitigation strategies
                4. Testing recommendations
                """,
                "Create Technical Documentation": f"""
                Create comprehensive technical documentation based on this JIRA issue:
                
                {formatted_issue}
                
                Include:
                1. Technical requirements
                2. Implementation details
                3. Dependencies and assumptions
                4. Acceptance criteria
                """,
                "Identify Similar Issues": f"""
                Analyze this JIRA issue and find similar patterns in the documentation:
                
                {formatted_issue}
                
                Identify:
                1. Similar issues or patterns
                2. Common solutions that were applied
                3. Lessons learned
                4. Prevention strategies
                """,
                "Extract Requirements": f"""
                Extract and organize requirements from this JIRA issue:
                
                {formatted_issue}
                
                Categorize into:
                1. Functional requirements
                2. Non-functional requirements
                3. Business requirements
                4. Technical constraints
                """,
                "Risk Assessment": f"""
                Perform a risk assessment for this JIRA issue:
                
                {formatted_issue}
                
                Assess:
                1. Technical risks
                2. Business impact
                3. Implementation complexity
                4. Recommended priority level
                """,
            }

            prompt = prompts.get(
                analysis_type, prompts["Generate Solution Suggestions"]
            )

            # Use your existing RAG system to analyze
            # This would integrate with your existing chat_handler or similar component
            result = {
                "issue_key": issue_key,
                "summary": issue["fields"].get("summary", "No summary"),
                "analysis": f"Analysis for {issue_key} - {analysis_type}",
                "prompt": prompt,
            }

            analysis_results.append(result)

        except Exception as e:
            st.error(f"Error analyzing {issue_key}: {str(e)}")

        progress_bar.progress((i + 1) / len(selected_issues))

    status_text.text("Analysis complete!")

    # Display results
    st.subheader("📊 Analysis Results")
    for result in analysis_results:
        with st.expander(
            f"🎫 {result['issue_key']}: {result['summary']}", expanded=True
        ):
            st.write("**Analysis Type:**", analysis_type)

            # Here you would typically call your RAG system with the prompt
            st.info("Integration point: This prompt would be sent to your RAG system")
            with st.expander("View Generated Prompt", expanded=False):
                st.code(result["prompt"])


def display_sidebar(components, settings):
    """Renders the entire sidebar for document management, JIRA, and stats."""
    with st.sidebar:
        st.header("🔍 Enhanced RAG System")
        st.write("*Advanced Retrieval-Augmented Generation*")
        st.write(
            "Ask questions about your documents, analyze JIRA tickets, or get insights from both!"
        )

        # Token Usage
        daily_tokens = components["query_ops"].get_todays_total_tokens()
        token_limit = 500000
        progress = min(daily_tokens / token_limit, 1.0)
        st.write("**Daily Token Usage**")
        st.progress(progress)
        st.caption(
            f"{daily_tokens:,} / {token_limit:,} tokens used ({progress*100:.1f}%)"
        )
        if progress > 0.8:
            st.warning("⚠️ Approaching daily limit!")

        # --- Model and Powering Info ---
        if settings.TEST == "True":
            powered_by = "Powered by Groq and Llama 4 Scout"
        else:
            powered_by = "Powered by Sanctuary and Claude 3.5 Sonnet"

        st.caption(powered_by)

        st.divider()

        # JIRA Connection
        display_jira_login()

        # JIRA Interface (moved from main area)
        if st.session_state.get("jira_authenticated", False):
            display_sidebar_jira_interface(components, settings)

        # Show context info
        context_info = []
        user_docs = components["doc_ops"].get_user_documents(st.session_state.user_id)
        if user_docs:
            context_info.append(f"📄 {len(user_docs)} documents")

        if st.session_state.get("jira_authenticated", False):
            issues_count = len(st.session_state.get("jira_issues", []))
            selected_count = len(st.session_state.get("jira_selected_issues", {}))
            if selected_count > 0:
                context_info.append(f"🎫 {selected_count} JIRA issues selected")
            elif issues_count > 0:
                context_info.append(f"🎫 {issues_count} JIRA issues loaded")

        if context_info:
            st.write("**Available Context:** " + " | ".join(context_info))

        st.divider()

        # File Uploader
        st.header("📁 Document Management")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=settings.SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            help=f"Max file size: {settings.MAX_UPLOAD_SIZE}MB",
        )

        if uploaded_files:
            process_files = st.button(
                "Process All Files",
                type="primary",
            )
            if process_files:
                progress_bar = st.progress(0)
                for i, uploaded_file in enumerate(uploaded_files):
                    file_ext = uploaded_file.name.split(".")[-1].lower()
                    if file_ext in [
                        "png",
                        "jpg",
                        "jpeg",
                        "tiff",
                        "tif",
                        "bmp",
                        "gif",
                        "webp",
                    ]:
                        st.info(f"🖼️ {uploaded_file.name} will be processed with OCR")
                    elif file_ext == "pdf":
                        st.info(
                            f"📄 {uploaded_file.name} will be processed with text extraction + OCR for images"
                        )
                    process_document_upload(
                        uploaded_file, components, settings, st.session_state.user_id
                    )
                    progress_bar.progress((i + 1) / len(uploaded_files))
                st.rerun()

        # Document List
        st.subheader("📄 Your Documents")
        try:
            user_docs = components["doc_ops"].get_user_documents(
                st.session_state.user_id
            )
            if user_docs:
                st.metric("Total Documents", len(user_docs))

                # Display documents with delete functionality
                for i, doc in enumerate(user_docs[:10]):
                    col1, col2 = st.columns([6, 1])

                    with col1:
                        status = "✅" if doc["processed"] else "⏳"
                        emoji = get_file_type_emoji(doc.get("file_type", ""))
                        size = format_file_size(doc.get("file_size", 0))

                        # Create a nice document card layout
                        st.markdown(
                            f"""
                        <div style="
                            background-color: var(--background-color);
                            border: 1px solid var(--border-color);
                            border-radius: 8px;
                            padding: 12px;
                            margin: 4px 0;
                            display: flex;
                            align-items: center;
                            gap: 8px;
                        ">
                            <span style="font-size: 1.2em;">{status} {emoji}</span>
                            <div>
                                <strong>{doc['document_name']}</strong><br>
                                <small style="color: var(--text-color-secondary);">{size}</small>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    with col2:
                        # Delete button
                        if st.button("🗑️", key=f"delete_{i}", help="Delete document"):
                            # Store the document ID to delete in session state
                            st.session_state.doc_to_delete = doc.get(
                                "id", doc.get("document_id")
                            )
                            st.rerun()

                # Handle deletion confirmation
                if hasattr(st.session_state, "doc_to_delete"):
                    st.warning("⚠️ Are you sure you want to delete this document?")
                    col1, col2, col3 = st.columns([1, 1, 1])

                    with col1:
                        if st.button("✅ Yes", key="confirm_delete_document"):
                            try:
                                # Call your delete function here
                                components["doc_ops"].delete_document(
                                    st.session_state.doc_to_delete
                                )
                                weav_manager.delete_chunks_by_document(
                                    document_id=st.session_state.doc_to_delete,
                                    user_id=st.session_state.user_info.get("email"),
                                )

                                del st.session_state.doc_to_delete
                                time.sleep(1)
                                st.toast("Document deleted successfully!")
                                time.sleep(0.2)

                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting document: {e}")

                    with col2:
                        if st.button("❌ No", key="cancel_delete_document"):
                            del st.session_state.doc_to_delete
                            st.rerun()

                if len(user_docs) > 10:
                    st.caption(f"... and {len(user_docs) - 10} more documents")
            else:
                st.info("No documents uploaded yet.")
        except Exception as e:
            st.error(f"Error loading documents: {e}")


def create_analysis_prompt(selected_issues, analysis_type):
    """Create analysis prompt for selected JIRA issues."""
    connector = st.session_state.jira_connector

    analysis_prompts = {
        "Generate Solutions": "Provide detailed solution suggestions, root cause analysis, and step-by-step approaches for resolving these issues.",
        "Create Documentation": "Generate comprehensive technical documentation covering requirements, implementation details, and acceptance criteria.",
        "Find Similar Issues": "Identify patterns, similar issues, and successful solution approaches from past experience.",
        "Extract Requirements": "Extract and organize functional, non-functional, and business requirements from these issues.",
        "Risk Assessment": "Perform risk assessment covering technical risks, business impact, and implementation challenges.",
    }

    prompt = f"""
Please {analysis_prompts.get(analysis_type, analysis_prompts['Generate Solutions'])}

JIRA Issues to Analyze:

"""

    for issue_key, issue in selected_issues.items():
        try:
            # Get detailed issue information
            detailed_issue = connector.get_issue_details(issue_key)
            comments = connector.get_issue_comments(issue_key)

            # Format issue for analysis
            formatted_issue = format_issue_for_rag(detailed_issue, comments)
            prompt += f"{formatted_issue}\n\n---\n\n"

        except Exception as e:
            # Fallback to basic issue info
            fields = issue.get("fields", {})
            summary = fields.get("summary", "No summary")
            description = fields.get("description", "No description")
            status = fields.get("status", {}).get("name", "Unknown")
            priority = fields.get("priority", {}).get("name", "Unknown")

            prompt += f"""
Issue: {issue_key}
Summary: {summary}
Status: {status}
Priority: {priority}
Description: {description}

---

"""

    return prompt


def display_chat_messages(model_name):
    """Displays the chat message history."""
    # with st.expander("Historical chat messages"):
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                display_sources(message.get("sources", []))
                time_info = f"{message.get('processing_time', 0):.2f}"
                tokens_info = f"{message.get('tokens_used', 0):,}"
                st.caption(
                    f"🧠 Model: {model_name} | ⏱️ Response time: {time_info}s | 🔢 Tokens: {tokens_info}"
                )


def display_user_info_sidebar(components):
    """Display user information in sidebar."""
    if hasattr(st.session_state, "user_info") and st.session_state.user_info:
        with st.sidebar:
            st.divider()

            st.subheader("👤 User Profile")

            user_info = st.session_state.user_info

            # Display user basic info
            st.write(f"**Email:** {user_info.get('email', 'Unknown')}")
            if user_info.get("display_name"):
                st.write(f"**Name:** {user_info['display_name']}")

            # Display user stats
            activity_stats = st.session_state.components[
                "user_ops"
            ].get_user_activity_stats(st.session_state.user_id)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", activity_stats["document_count"])
                st.metric("Total Queries", activity_stats["total_queries"])
            with col2:
                st.metric("Today's Tokens", activity_stats["today_tokens"])
                st.metric("Total Tokens", activity_stats["total_tokens"])

            # Initialize session state for delete confirmation
            if "show_delete_confirmation" not in st.session_state:
                st.session_state.show_delete_confirmation = False

            del_queries = st.button("Delete all Queries", type="primary")
            if del_queries:
                st.session_state.show_delete_confirmation = True

            # Show confirmation dialog if needed
            if st.session_state.show_delete_confirmation:
                st.warning("⚠️ Are you sure you want to delete all your queries?")
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    if st.button(
                        "Delete all queries",
                        key="confirm_delete_queries",
                        type="primary",
                    ):
                        try:
                            print(st.session_state.user_id)
                            # Call your delete function here
                            components["query_ops"].delete_user_queries(
                                st.session_state.user_id
                            )
                            st.toast("All queries cleared!")
                            # Reset the confirmation state
                            st.session_state.show_delete_confirmation = False
                            st.session_state.messages = []
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting queries: {e}")
                            st.session_state.show_delete_confirmation = False

                with col2:
                    if st.button("Back to safety", key="cancel_delete_queries"):
                        # Reset the confirmation state
                        st.session_state.show_delete_confirmation = False
                        st.rerun()

            # Show recent activity
            if activity_stats["recent_activity"]:
                st.write("**Recent Activity (7 days):**")
                for date, count in activity_stats["recent_activity"][
                    :3
                ]:  # Show last 3 days
                    st.write(f"• {date}: {count} queries")

            first_login = user_info.get("first_login", "Unknown")
            if isinstance(first_login, datetime):
                first_login_str = first_login.strftime("%Y-%m-%d")
            elif first_login != "Unknown":
                first_login_str = str(first_login)[:10]
            else:
                first_login_str = "Unknown"

            st.write(f"**Member since:** {first_login_str}")
