import streamlit as st


def authenticate():
    # Authentication check
    if not st.user.is_logged_in:
        col1, col2, col3 = st.columns(3)
        with col2:
            with st.container(
                border=True,
                horizontal_alignment="center",
            ):
                st.write(
                    "Welcome to team Cosmic's Space Force AI Challenge JIRA-RAG tool! To continue, please log in."
                )
                st.button("Log in", on_click=st.login, type="secondary")
                st.stop()

    # Provider (e.g. google, github)
    provider = st.user.get("sub").split("|")[0]

    # Inject CSS
    st.markdown(
        """
    <style>
        .profile-dropdown {
            position: fixed;
            top: 4rem; /* moved down further */
            right: 1rem;
            z-index: 2000;
        }
        .profile-img {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            cursor: pointer;
            display: block;
        }
        .dropdown-content {
            display: none;
            position: absolute;
            right: 0;
            top: 45px; /* positioned right below the image */
            background-color: #1F2324;
            min-width: 200px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
            padding: 16px;
            border-radius: 8px;
            font-size: 14px;
            border: 1px solid #e0e0e0;
        }
        .profile-dropdown:hover .dropdown-content {
            display: block;
        }
        /* Create a bridge to prevent hover gaps */
        .dropdown-content::before {
            content: '';
            position: absolute;
            top: -1rem; /* Larger gap coverage */
            left: 0; /* Cover the full width */
            right: 0;
            height: 1rem;
            background: transparent;
            pointer-events: auto;
        }
        .logout-button {
            margin-top: 12px;
            width: 100%;
            background: #f44336;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 13px;
            font-family: inherit;
        }
        .logout-button:hover {
            background: #d32f2f;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Use query params to handle logout
    query_params = st.query_params
    if query_params.get("logout") == "true":
        st.logout()  # This will clear session state and redirect
        st.stop()  # Prevent further execution

    # Render dropdown with profile + info + logout link
    st.markdown(
        f"""
    <div class="profile-dropdown">
        <img src="{st.user.get("picture")}" alt="Profile Picture" class="profile-img">
        <div class="dropdown-content">
            <div>Welcome, <strong>{st.user.get("name")}</strong></div>
            <div style="margin-top:6px; font-size:13px;">
                Logged in via {provider}
            </div>
            <a href="?logout=true" class="logout-button" style="text-decoration: none; display: block; text-align: center;">Log out</a>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
