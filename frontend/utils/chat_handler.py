# utils/chat_handler.py

import json
import time
import logging
import traceback
import streamlit as st

logger = logging.getLogger(__name__)


def load_chat_history(query_ops, user_id):
    """Load all queries from the database and format them for display."""
    if st.session_state.get("messages_loaded", False):
        return
    try:
        st.session_state.messages = []
        user_queries = query_ops.get_user_queries(limit=100, user_id=user_id)
        for query_data in user_queries:
            st.session_state.messages.append(
                {"role": "user", "content": query_data["user_query"]}
            )

            raw_sources = query_data.get("answer_sources_used", "[]")
            sources = (
                json.loads(raw_sources) if isinstance(raw_sources, str) else raw_sources
            )
            assistant_message = {
                "role": "assistant",
                "content": query_data["answer_text"],
                "sources": sources,
                "processing_time": query_data.get("processing_time"),
                "tokens_used": query_data.get("tokens_used", 0),
            }

            # Check if this was a JIRA-enhanced query (you might want to store this in DB)
            # For now, we'll detect it from the content
            if any(
                keyword in query_data["user_query"].lower()
                for keyword in ["jira", "ticket", "issue", "analyze"]
            ):
                assistant_message["jira_context_used"] = True

            st.session_state.messages.append(assistant_message)

        st.session_state.messages_loaded = True
    except Exception as e:
        error_msg = f"Failed to load chat history: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(f"{error_msg}\n\n**Traceback:**\n```\n{traceback.format_exc()}\n```")


def handle_user_query(query, components, model_name):
    """Processes a user query, generates a response, and updates the state."""
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.info("🔍 Searching knowledge base...")

        try:
            start_time = time.time()
            answer, sources, _, total_tokens = components[
                "retriever"
            ].retrieve_and_generate(query, st.session_state.user_id)
            total_elapsed = time.time() - start_time

            status_placeholder.empty()

            # Display response and save to DB
            st.write(answer)
            assistant_message = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "processing_time": total_elapsed,
                "tokens_used": total_tokens,
            }
            st.session_state.messages.append(assistant_message)

            components["query_ops"].insert_query(
                user_query=query,
                answer_text=answer,
                answer_sources=sources,
                user_id=st.session_state.user_id,
                processing_time=total_elapsed,
                chunks_used=len(sources) if sources else 0,
                tokens_used=total_tokens,
            )
            st.rerun()

        except Exception as e:
            status_placeholder.empty()
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Query handling error: {traceback.format_exc()}")


def is_jira_related_query(query):
    """Check if a query is related to JIRA functionality."""
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
        "requirement",
        "documentation",
        "similar",
        "pattern",
        "resolution",
        "status",
        "priority",
    ]
    return any(keyword in query.lower() for keyword in jira_keywords)


def get_jira_context_summary():
    """Get a summary of available JIRA context."""
    if not st.session_state.get("jira_authenticated", False):
        return None

    selected_issues = st.session_state.get("jira_selected_issues", {})
    loaded_issues = st.session_state.get("jira_issues", [])

    context_info = {
        "authenticated": True,
        "selected_count": len(selected_issues),
        "loaded_count": len(loaded_issues),
        "has_context": len(selected_issues) > 0 or len(loaded_issues) > 0,
    }

    return context_info


def format_jira_enhanced_query(original_query, jira_context):
    """Format a query with JIRA context for better RAG processing."""
    enhanced_query = f"""
User Query: {original_query}

Context: The user has access to JIRA tickets and is asking a question that may relate to:
- Technical issue resolution
- Project management insights  
- Requirements analysis
- Risk assessment
- Solution recommendations

Please provide a comprehensive response that considers both the uploaded documents and any relevant JIRA context provided.

{jira_context}
"""
    return enhanced_query


def extract_analysis_type_from_query(query):
    """Extract the type of analysis requested from the user's query."""
    query_lower = query.lower()

    if any(
        word in query_lower
        for word in ["solution", "solve", "fix", "resolve", "how to"]
    ):
        return "Generate Solution Suggestions"
    elif any(
        word in query_lower
        for word in ["document", "documentation", "spec", "requirements"]
    ):
        return "Create Technical Documentation"
    elif any(
        word in query_lower
        for word in ["similar", "like", "pattern", "compare", "related"]
    ):
        return "Identify Similar Issues"
    elif any(
        word in query_lower
        for word in ["requirement", "functional", "non-functional", "criteria"]
    ):
        return "Extract Requirements"
    elif any(
        word in query_lower
        for word in ["risk", "impact", "assessment", "danger", "problem"]
    ):
        return "Risk Assessment"
    else:
        return "Generate Solution Suggestions"  # Default


def prepare_jira_analysis_prompt(selected_issues, analysis_type, user_query):
    """Prepare a comprehensive prompt for JIRA issue analysis."""

    analysis_prompts = {
        "Generate Solution Suggestions": """
Based on the JIRA issues and available documentation, provide:
1. Root cause analysis for each issue
2. Step-by-step solution approaches
3. Potential risks and mitigation strategies  
4. Testing and validation recommendations
5. Implementation timeline estimates
        """,
        "Create Technical Documentation": """
Generate comprehensive technical documentation covering:
1. Technical requirements and specifications
2. Implementation details and architecture
3. Dependencies, assumptions, and constraints
4. Acceptance criteria and success metrics
5. Maintenance and support guidelines
        """,
        "Identify Similar Issues": """
Analyze patterns and identify:
1. Similar issues or recurring problems
2. Common root causes across tickets
3. Successful solution patterns that were applied
4. Lessons learned and best practices
5. Prevention strategies for future occurrences
        """,
        "Extract Requirements": """
Extract and organize requirements into:
1. Functional requirements (what the system should do)
2. Non-functional requirements (performance, security, etc.)
3. Business requirements and objectives
4. Technical constraints and limitations
5. User experience and interface requirements
        """,
        "Risk Assessment": """
Perform comprehensive risk assessment covering:
1. Technical risks and complexity analysis
2. Business impact and stakeholder effects
3. Implementation challenges and dependencies
4. Resource requirements and timeline risks
5. Recommended priority levels and mitigation plans
        """,
    }

    prompt = f"""
User Query: {user_query}

Analysis Type: {analysis_type}

{analysis_prompts.get(analysis_type, analysis_prompts["Generate Solution Suggestions"])}

JIRA Issues for Analysis:
"""

    # Add each selected issue
    for issue_key, issue_data in selected_issues.items():
        fields = issue_data.get("fields", {})
        prompt += f"""
---
Issue: {issue_key}
Summary: {fields.get('summary', 'No summary')}
Status: {fields.get('status', {}).get('name', 'Unknown')}
Priority: {fields.get('priority', {}).get('name', 'Unknown')}
Description: {fields.get('description', 'No description')}
---
"""

    prompt += """

Please provide a detailed analysis based on the available documentation and the JIRA context above. 
Focus on actionable insights and practical recommendations.
"""

    return prompt
