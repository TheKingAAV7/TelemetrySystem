
import streamlit as st
from TelemetrySystem import TelemetrySystem



if "telemetry_system" not in st.session_state:
    st.session_state.telemetry_system = TelemetrySystem()

st.title("Telemetry Natural Language Query System")


user_query = st.text_input("Enter your query (natural language):", "")

if st.button("Submit") and user_query.strip():
    with st.spinner("Processing..."):
        success, result, generated_sql = st.session_state.telemetry_system.process_natural_language_query(user_query)

    st.subheader("Generated SQL")
    st.code(generated_sql or "No SQL generated", language="sql")

    if success:
        st.subheader("Query Results")
        st.dataframe(result)
    else:
        st.subheader("Error")
        st.error(result)
