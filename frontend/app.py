# app.py

import streamlit as st
import logging
import traceback
from datetime import datetime
import time

# Import project modules
from config.settings import settings
from utils.system_init import initialize_system
from utils.ui import display_sidebar, display_chat_messages, display_user_info_sidebar
from utils.chat_handler import load_chat_history, handle_user_query
from utils.jira_connector import format_issue_for_rag
from utils.authentication import authenticate

# --- Page Configuration and Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Enhanced RAG System with JIRA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Model and Powering Info ---
if settings.TEST == "True":
    model_name = "Llama 4 Scout"
else:
    model_name = "Claude 3.5 Sonnet"


def initialize_user_session():
    """Initialize user session and tracking."""
    # Authenticate user first
    authenticate()

    # Get user email and create/update user record
    user_email = st.user.get("email")
    user_name = st.user.get("name", "")

    if not user_email:
        st.error("Authentication required. Please log in.")
        st.stop()

    # Initialize system if not already done
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if not st.session_state.initialized:
        with st.spinner("Initializing Enhanced RAG System..."):
            st.session_state.components = initialize_system()
            st.session_state.initialized = True

    components = st.session_state.components

    # Create or update user in database
    try:
        user_id = components["user_ops"].create_or_update_user(user_email, user_name)
        st.session_state.user_id = user_id
        st.session_state.user_email = user_email
        st.session_state.user_name = user_name

        # Get user info for display
        user_info = components["user_ops"].get_user_info(user_id)
        st.session_state.user_info = user_info

    except Exception as e:
        logger.error(f"Error initializing user session: {e}")
        st.error("Error initializing user session. Please refresh the page.")
        st.stop()


# --- Session State Initialization ---
def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize JIRA-related session state
    if "jira_authenticated" not in st.session_state:
        st.session_state.jira_authenticated = False
    if "jira_connector" not in st.session_state:
        st.session_state.jira_connector = None
    if "jira_user_info" not in st.session_state:
        st.session_state.jira_user_info = {}
    if "jira_issues" not in st.session_state:
        st.session_state.jira_issues = []
    if "jira_selected_issues" not in st.session_state:
        st.session_state.jira_selected_issues = {}
    if "pending_jira_analysis" not in st.session_state:
        st.session_state.pending_jira_analysis = None


def handle_enhanced_user_query(query, components, model_name):
    """Enhanced query handler that can incorporate JIRA context."""

    # Check if this is a JIRA-related query or if JIRA context should be added
    jira_keywords = [
        "jira",
        "ticket",
        "issue",
        "bug",
        "story",
        "task",
        "epic",
        "analyze",
        "solution",
        "risk",
    ]
    is_jira_query = any(keyword in query.lower() for keyword in jira_keywords)

    # Prepare enhanced context
    enhanced_query = query
    jira_context_added = False

    # Add JIRA context if available and relevant
    if st.session_state.get("jira_authenticated", False):
        selected_issues = st.session_state.get("jira_selected_issues", {})
        loaded_issues = st.session_state.get("jira_issues", [])

        # If user has selected specific issues, prioritize those
        if selected_issues and is_jira_query:
            jira_context = "\n\n--- JIRA CONTEXT ---\n"
            jira_context += f"Selected JIRA Issues ({len(selected_issues)}):\n\n"

            connector = st.session_state.get("jira_connector")
            for issue_key, issue in selected_issues.items():
                try:
                    # Get detailed issue information
                    detailed_issue = (
                        connector.get_issue_details(issue_key) if connector else issue
                    )
                    comments = (
                        connector.get_issue_comments(issue_key) if connector else []
                    )

                    # Format issue for RAG
                    formatted_issue = format_issue_for_rag(detailed_issue, comments)
                    jira_context += f"{formatted_issue}\n\n---\n\n"

                except Exception as e:
                    logger.error(f"Error fetching details for {issue_key}: {e}")
                    # Fallback to basic issue info
                    fields = issue.get("fields", {})
                    summary = fields.get("summary", "No summary")
                    jira_context += f"Issue {issue_key}: {summary}\n\n"

            enhanced_query = query + jira_context
            jira_context_added = True

        # If no specific issues selected but user mentions JIRA, provide general context
        elif loaded_issues and is_jira_query and not jira_context_added:
            jira_context = f"\n\n--- AVAILABLE JIRA ISSUES ---\n"
            jira_context += f"You have {len(loaded_issues)} JIRA issues loaded. Here's a summary:\n\n"

            for issue in loaded_issues[:5]:  # Show first 5 as context
                key = issue["key"]
                fields = issue.get("fields", {})
                summary = fields.get("summary", "No summary")
                status = fields.get("status", {}).get("name", "Unknown")
                priority = fields.get("priority", {}).get("name", "Unknown")

                jira_context += f"- {key}: {summary}\n"
                jira_context += f"  Status: {status}, Priority: {priority}\n\n"

            if len(loaded_issues) > 5:
                jira_context += f"... and {len(loaded_issues) - 5} more issues.\n\n"

            enhanced_query = query + jira_context
            jira_context_added = True

    # Add a note about JIRA context if it was added
    if jira_context_added:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
            st.caption("🔗 Including JIRA context in analysis")
    else:
        # Use the original handler for non-JIRA queries
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

    # Process the enhanced query
    with st.chat_message("assistant"):
        status_placeholder = st.empty()

        if jira_context_added:
            status_placeholder.info("🔍 Analyzing with JIRA context...")
        else:
            status_placeholder.info("🔍 Searching knowledge base...")

        try:
            import time

            start_time = time.time()

            # Use your existing retrieval system with the enhanced query
            answer, sources, _, total_tokens = components[
                "retriever"
            ].retrieve_and_generate(enhanced_query, st.session_state.user_id)
            total_elapsed = time.time() - start_time

            status_placeholder.empty()

            # Display response
            st.write(answer)

            # Prepare assistant message
            assistant_message = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "processing_time": total_elapsed,
                "tokens_used": total_tokens,
            }

            # Add JIRA context indicator if used
            if jira_context_added:
                assistant_message["jira_context_used"] = True
                selected_count = len(st.session_state.get("jira_selected_issues", {}))
                if selected_count > 0:
                    assistant_message["jira_issues_analyzed"] = selected_count

            st.session_state.messages.append(assistant_message)

            # Save to database (store original query, not enhanced)
            components["query_ops"].insert_query(
                user_query=query,  # Store original query
                answer_text=answer,
                answer_sources=sources,
                user_id=st.session_state.user_id,
                processing_time=total_elapsed,
                chunks_used=len(sources) if sources else 0,
                tokens_used=total_tokens,
            )

            # Update user statistics
            try:
                components["user_ops"].update_user_stats(
                    st.session_state.user_id, increment_queries=1
                )
            except Exception as e:
                logger.error(f"Error updating user stats: {e}")

            st.rerun()

        except Exception as e:
            status_placeholder.empty()
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Enhanced query handling error: {traceback.format_exc()}")


def display_jira_status_info():
    """Display JIRA connection status and quick stats."""
    if st.session_state.get("jira_authenticated", False):
        user_info = st.session_state.get("jira_user_info", {})
        issues_count = len(st.session_state.get("jira_issues", []))
        selected_count = len(st.session_state.get("jira_selected_issues", {}))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("JIRA Status", "🟢 Connected")
        with col2:
            display_name = user_info.get("displayName", "Unknown")
            if len(display_name) > 15:
                display_name = display_name[:15] + "..."
            st.metric("User", display_name)
        with col3:
            st.metric("Issues Loaded", issues_count)
        with col4:
            st.metric("Selected for Analysis", selected_count)


def main():
    """Main application flow."""

    # Initialize user session and authentication
    initialize_user_session()

    # Initialize other session state variables
    init_session_state()

    # Get components
    components = st.session_state.components
    # --- Sidebar ---
    display_sidebar(components, settings)

    # Display user info in sidebar
    display_user_info_sidebar(components)

    # --- Load Chat History ---
    load_chat_history(components["query_ops"], user_id=st.session_state.user_id)

    # --- Display Chat Messages ---
    display_chat_messages(model_name)

    # --- Handle Pending JIRA Analysis ---
    if st.session_state.get("pending_jira_analysis"):
        pending = st.session_state.pending_jira_analysis
        del st.session_state.pending_jira_analysis

        # Process the analysis
        handle_enhanced_user_query(pending["prompt"], components, model_name)

    # --- User Input with Enhanced Capabilities ---
    placeholder_text = "Ask anything about your documents"
    if st.session_state.get("jira_authenticated", False):
        selected_count = len(st.session_state.get("jira_selected_issues", {}))
        issues_count = len(st.session_state.get("jira_issues", []))

        if selected_count > 0:
            placeholder_text = f"Ask about your documents or analyze {selected_count} selected JIRA issues..."
        elif issues_count > 0:
            placeholder_text = (
                f"Ask about your documents or {issues_count} loaded JIRA issues..."
            )
        else:
            placeholder_text = "Ask about your documents or JIRA tickets (search for issues in sidebar)..."

    if query := st.chat_input(placeholder_text):
        # Check if user has documents or JIRA access
        has_jira = st.session_state.get("jira_authenticated", False)
        user_docs = components["doc_ops"].get_user_documents(st.session_state.user_id)

        if not user_docs and not has_jira:
            st.warning("⚠️ Please upload some documents or connect to JIRA first!")
        elif not user_docs and has_jira:
            # Allow JIRA-only queries
            handle_enhanced_user_query(query, components, model_name)
        else:
            # Handle normally with enhanced JIRA context
            handle_enhanced_user_query(query, components, model_name)

    if st.session_state.user_email in [
        "smaueltown@gmail.com",
    ]:
        with st.sidebar:
            with st.expander("🔧 Admin Panel"):
                st.subheader("User Management")

                if st.button("View All Users"):
                    users_summary = components["user_ops"].get_all_users_summary()
                    st.write("**All Users Summary:**")
                    for user in users_summary:
                        st.write(
                            f"• {user['email']} - {user['total_queries']} queries, {user['total_tokens']} tokens"
                        )


if __name__ == "__main__":
    main()
