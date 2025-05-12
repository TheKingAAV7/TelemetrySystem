import duckdb
import random
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_core.prompts import PromptTemplate
import pandas as pd
from sqlalchemy import create_engine, text
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelemetryDB:
    def __init__(self, db_path: str = 'telemetry.duckdb'):
        self.db_path = db_path
        try:
            self.engine = create_engine(f"duckdb:///{db_path}")
            self._init_db()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise RuntimeError(f"Could not initialize database. Ensure the database file '{db_path}' is accessible and not corrupted.")
    
    def _init_db(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp TIMESTAMP,
                server_id TEXT,
                service_name TEXT,
                cpu_usage DOUBLE,
                memory_usage DOUBLE,
                disk_usage DOUBLE
            )
            """))
            conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON metrics (timestamp)
            """))
            conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_server_id ON metrics (server_id)
            """))
            conn.commit()
        logger.info("Database initialized with indexes")
    
    def insert_sample_data(self, num_rows: int = 1000) -> None:
        servers = ["server1", "server2", "server3", "server4", "server5"]
        services = ["auth", "billing", "analytics", "api", "database"]
        now = datetime.now()
        rows = []
        
        for _ in range(num_rows):
            timestamp = now - timedelta(minutes=random.randint(0, 10080))
            server = random.choice(servers)
            service = random.choice(services)
            cpu = round(random.uniform(10, 100), 2)
            memory = round(random.uniform(20, 100), 2)
            disk = round(random.uniform(30, 99.5), 2)
            
            if random.random() < 0.1:
                disk = round(random.uniform(90, 100), 2)
            if random.random() < 0.15:
                memory = round(random.uniform(65, 100), 2)
            
            rows.append((timestamp, server, service, cpu, memory, disk))
        
        with self.engine.connect() as conn:
            # Use executemany for better performance with multiple rows
            conn.execute(
                text("""
                INSERT INTO metrics (timestamp, server_id, service_name, cpu_usage, memory_usage, disk_usage)
                VALUES (:timestamp, :server_id, :service_name, :cpu_usage, :memory_usage, :disk_usage)
                """),
                [
                    {
                        "timestamp": row[0],
                        "server_id": row[1],
                        "service_name": row[2],
                        "cpu_usage": row[3],
                        "memory_usage": row[4],
                        "disk_usage": row[5]
                    }
                    for row in rows
                ]
            )
            conn.commit()
        logger.info(f"Inserted {num_rows} sample records")
    
    def execute_query(self, query: str) -> Optional[pd.DataFrame]:
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql_query(text(query), conn)
                return result
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None
    
    def get_query_result(self, query: str) -> Tuple[bool, Any]:
        result = self.execute_query(query)
        if result is not None:
            return True, result
        return False, "Query execution failed"
    
    def run(self, query: str, fetch: str = "all") -> Tuple[bool, Any]:
        """Execute a query and return results."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                if fetch == "all":
                    data = result.fetchall()
                    return True, data
                elif fetch == "one":
                    data = result.fetchone()
                    return True, data
                return True, None
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return False, str(e)
    
    def _execute(self, query: str):
        """Execute a query and return the cursor for metadata access."""
        try:
            with self.engine.connect() as conn:
                return conn.execute(text(query))
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise e
    
    def close(self) -> None:
        self.engine.dispose()
        logger.info("Database engine disposed")

class QueryProcessor:
    def __init__(self, db: TelemetryDB, model: str = 'mistral'):
        try:
            self.llm = Ollama(model=model, base_url="http://localhost:11434")
            logger.info("Ollama initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            raise RuntimeError("Could not initialize LLM. Ensure Ollama is running and compatible with LangChain.")
        self.db = db
        self.sql_db = SQLDatabase(self.db.engine)
        self._init_chain()
    
    def _init_chain(self) -> None:
        prompt_template = """
        You are a SQL expert for observability data. Convert the natural language query into a DuckDB-compatible SQL query.
        Table: metrics (timestamp TIMESTAMP, server_id TEXT, service_name TEXT, cpu_usage DOUBLE, memory_usage DOUBLE, disk_usage DOUBLE).
        Note: cpu_usage, memory_usage, and disk_usage are in percentage units (0 to 100).

        Rules:
        1. Return ONLY the SQL query, without any prefixes (e.g., 'SQLQuery:'), explanations, or additional text
        2. Use explicit column names in SELECT
        3. Use INTERVAL for time ranges, e.g., 'CAST(now() AS TIMESTAMP) - INTERVAL 24 HOUR'
        4. Ensure queries are read-only (SELECT only)
        5. Use DuckDB-compatible datetime functions
        6. Cast now() to TIMESTAMP when comparing with timestamp column, e.g., CAST(now() AS TIMESTAMP)
        7. For percentage thresholds (e.g., >65%), compare directly with the column value, e.g., memory_usage > 65
        8. Do not use window functions (e.g., OVER) in WHERE clauses; use subqueries or CTEs if needed
        9. Avoid unnecessary subqueries for simple threshold queries

        Example:
        Input: Which servers had >65% memory usage in the last 24 hours?
        Output: SELECT server_id FROM metrics WHERE memory_usage > 65 AND timestamp >= CAST(now() AS TIMESTAMP) - INTERVAL '24 HOUR'

        Input: {input}
        Output:
        """
        custom_prompt = PromptTemplate.from_template(prompt_template)
        
        self.sql_chain = SQLDatabaseChain.from_llm(
            llm=self.llm,
            db=self.sql_db,
            prompt=custom_prompt,
            verbose=True,
            return_intermediate_steps=True
        )
    
    def process_query(self, user_query: str) -> Tuple[bool, Any, str]:
        try:
            result = self.sql_chain(user_query)
            generated_sql = result["intermediate_steps"][1]  # Get the SQL query from steps
            generated_sql = re.sub(
                r'^(?:Query|SQLQuery|SQL|Explanation|Note):.*?\n|```sql\s*|\s*```|Explanation:.*$|Note:.*$',
                '',
                generated_sql,
                flags=re.MULTILINE | re.DOTALL
            ).strip()
            
            if not self.validate_sql(generated_sql):
                return False, "Generated SQL failed safety validation", generated_sql
            
            success, query_result = self.db.run(generated_sql, fetch="all")
            if success:
                if query_result:
                    try:
                        cursor = self.db._execute(generated_sql)
                        columns = [desc[0] for desc in cursor.description]
                        df = pd.DataFrame(query_result, columns=columns)
                        return True, df, generated_sql
                    except Exception as e:
                        # If we can't get column names, create a dataframe without them
                        df = pd.DataFrame(query_result)
                        return True, df, generated_sql
                return True, pd.DataFrame(), generated_sql
            error_msg = str(query_result)
            self._add_error_suggestions(error_msg)
            return False, error_msg, generated_sql
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_msg = str(e)
            self._add_error_suggestions(error_msg)
            return False, error_msg, ""
    
    def _add_error_suggestions(self, error_msg: str) -> str:
        """Add helpful suggestions based on error message."""
        if "Binder Error: WHERE clause cannot contain window functions" in error_msg:
            error_msg += "\nSuggestion: Rewrite the query to avoid window functions in WHERE. Use a subquery or CTE for window calculations."
        elif "Binder Error: Cannot compare values of type TIMESTAMP" in error_msg:
            error_msg += "\nSuggestion: Ensure the SQL query casts now() to TIMESTAMP, e.g., CAST(now() AS TIMESTAMP)."
        elif "Parser Error: syntax error" in error_msg:
            error_msg += "\nSuggestion: Ensure the SQL query contains only valid SQL syntax without prefixes or explanations."
        return error_msg
    
    def validate_sql(self, sql: str) -> bool:
        dangerous_patterns = [
            r'DROP\s+TABLE',
            r'DELETE\s+FROM',
            r'TRUNCATE\s+TABLE',
            r'ALTER\s+TABLE',
            r'INSERT\s+INTO',
            r'UPDATE\s+.*\s+SET',
            r'EXEC\s+',
            r'OVER\s*\([^)]*\)\s*[^;]*\s*WHERE'
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                logger.warning(f"Invalid SQL detected: {pattern}")
                return False
        return True

class ObservabilityAssistant:
    def __init__(self, db_path: str = 'telemetry.duckdb', model: str = 'mistral'):
        self.db = TelemetryDB(db_path)
        self.query_processor = QueryProcessor(self.db, model)
    
    def setup_sample_data(self, num_rows: int = 1000) -> None:
        sample_query = "SELECT COUNT(*) FROM metrics"
        success, result = self.db.get_query_result(sample_query)
        if success and result.iloc[0, 0] == 0:
            logger.info("Database is empty. Generating sample data...")
            self.db.insert_sample_data(num_rows)
        elif not success:
            logger.warning("Failed to check if database is empty. Assuming it needs sample data.")
            self.db.insert_sample_data(num_rows)
    
    def process_query(self, query: str) -> Tuple[bool, Any, str]:
        return self.query_processor.process_query(query)
    
    def close(self) -> None:
        self.db.close()

def main():
    st.title("Observability Assistant")
    st.write("Ask questions about server metrics (CPU, memory, disk usage) in natural language.")
    
    if 'assistant' not in st.session_state:
        try:
            st.session_state['assistant'] = ObservabilityAssistant()
            st.session_state['assistant'].setup_sample_data()
        except RuntimeError as e:
            st.error(f"Initialization failed: {e}")
            return
    
    sample_queries = [
        "Which servers had >65% memory usage in the last 24 hours?",
        "Did any service spike over 85% CPU last week?",
        "List hosts with >90% disk usage in the past 12 hours",
    ]
    
    st.subheader("Try these sample queries:")
    for sq in sample_queries:
        if st.button(sq):
            st.session_state['query'] = sq
    
    query = st.text_input("Enter your query:", value=st.session_state.get('query', ''))
    
    if st.button("Run Query") and query:
        with st.spinner("Processing query..."):
            success, result, generated_sql = st.session_state['assistant'].process_query(query)
            
            st.subheader("Generated SQL")
            st.code(generated_sql, language="sql")
            
            if success:
                st.subheader("Query Results")
                if isinstance(result, pd.DataFrame) and not result.empty:
                    st.dataframe(result)
                else:
                    st.write("No results found.")
            else:
                st.error(f"Error: {result}")
    
    # Don't close the assistant here - it's stored in session_state

if __name__ == "__main__":
    main()