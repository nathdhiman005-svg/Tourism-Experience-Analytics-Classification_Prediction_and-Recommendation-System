import streamlit as st
import db

def render():
    conn = db.get_db_connection()
    
    st.title("🔐 Welcome to Tourism Experience Analytics")
    st.markdown("Please log in or sign up to access your personalized portal.")
    
    # Create tabs for Log In and Sign Up
    tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])
    
    with tab_login:
        st.subheader("Log In to Your Account")
        with st.form("login_form"):
            login_email = st.text_input("Email")
            login_password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Log In")
            
            if submit_login:
                if not login_email or not login_password:
                    st.error("Please provide both email and password.")
                else:
                    success, role = db.authenticate_user(conn, login_email, login_password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.role = role
                        st.session_state.user_email = login_email
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
                        
    with tab_signup:
        st.subheader("Create a New Account")
        with st.form("signup_form"):
            signup_username = st.text_input("Username")
            signup_email = st.text_input("Email")
            signup_password = st.text_input("Password", type="password")
            submit_signup = st.form_submit_button("Sign Up")
            
            if submit_signup:
                if not signup_username or not signup_email or not signup_password:
                    st.error("Please fill out all fields.")
                else:
                    # Hardcoded rule: admin@admin.com gets the Admin role
                    role = "Admin" if signup_email.lower() == "admin@admin.com" else "User"
                    
                    success, msg = db.create_user(conn, signup_username, signup_email, signup_password, role)
                    
                    if success:
                        st.success(f"Account created successfully for {signup_email}! You can now switch to the Log In tab.")
                    else:
                        st.error(f"Sign Up failed: {msg}")
