import duckdb
import random
from datetime import datetime, timedelta
import logging
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TelemetryDB:
    def __init__(self, db_path: str = ':memory:'):
        self.conn = duckdb.connect(database=db_path)
        self._init_db()
    
    def _init_db(self):
        self.conn.execute("""
        CREATE TABLE metrics (
            timestamp TIMESTAMP,
            server_id TEXT,
            service_name TEXT,
            cpu_usage DOUBLE,
            memory_usage DOUBLE,
            disk_usage DOUBLE
        )
        """)
        logger.info("Database initialized")
    
    def insert_sample_data(self, num_rows: int = 100):
        servers = ["server1", "server2", "server3"]
        services = ["auth", "billing", "api"]
        now = datetime.now()
        rows = []
        
        for _ in range(num_rows):
            timestamp = now - timedelta(minutes=random.randint(0, 1440))  # last 24 hours
            server = random.choice(servers)
            service = random.choice(services)
            cpu = round(random.uniform(10, 100), 2)
            memory = round(random.uniform(20, 90), 2)
            disk = round(random.uniform(30, 90), 2)
            rows.append((timestamp, server, service, cpu, memory, disk))
        
        self.conn.executemany("""
            INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?)
        """, rows)
        logger.info(f"Inserted {num_rows} sample records")
    
    def execute_query(self, query: str):
        try:
            return True, self.conn.execute(query).fetchdf()
        except Exception as e:
            logger.error(f"Query error: {e}")
            return False, str(e)
    
    def close(self):
        self.conn.close()

class QueryCache:
    def __init__(self, model: str = 'mistral'):
        self.embeddings = OllamaEmbeddings(model=model)
        # Initialize with some examples
        self.examples = [
            {
                "query": "Show servers with CPU > 80% in last hour",
                "sql": "SELECT server_id, cpu_usage FROM metrics WHERE cpu_usage > 80 AND timestamp >= CAST(now() AS TIMESTAMP) - INTERVAL '1 HOUR'"
            },
            {
                "query": "List services with memory > 70% today",
                "sql": "SELECT service_name, memory_usage FROM metrics WHERE memory_usage > 70 AND timestamp >= CAST(now() AS TIMESTAMP) - INTERVAL '24 HOUR'"
            },
            {
                "query": "Find servers with disk > 85% in last 6 hours",
                "sql": "SELECT server_id, disk_usage FROM metrics WHERE disk_usage > 85 AND timestamp >= CAST(now() AS TIMESTAMP) - INTERVAL '6 HOUR'"
            }
        ]
        self._init_vector_db()
    
    def _init_vector_db(self):
        texts = [example["query"] for example in self.examples]
        metadatas = [{"sql": example["sql"]} for example in self.examples]
        try:
            self.vector_db = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            logger.info("Vector DB initialized with example queries")
        except Exception as e:
            logger.error(f"Vector DB initialization error: {e}")
            self.vector_db = None
    
    def find_similar(self, query: str, k: int = 3):
        if not self.vector_db:
            return []
        
        try:
            docs = self.vector_db.similarity_search(query, k=k)
            return [(doc.page_content, doc.metadata["sql"]) for doc in docs]
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return []
    
    def add_query(self, query: str, sql: str):
        if not self.vector_db:
            return
        
        try:
            self.vector_db.add_texts([query], [{"sql": sql}])
            logger.info(f"Added query to vector DB: {query}")
        except Exception as e:
            logger.error(f"Error adding to vector DB: {e}")

class RAGPipeline:
    def __init__(self, model: str = 'mistral'):
        self.llm = Ollama(model=model)
        self.query_cache = QueryCache(model=model)
        self.base_prompt = """
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

        
        """
    
    def generate_sql(self, question: str):
        similar_queries = self.query_cache.find_similar(question)
        
        prompt_template = self.base_prompt
        
        # Add similar examples if found
        if similar_queries:
            prompt_template += "\n\nSimilar queries for reference:"
            for idx, (similar_q, similar_sql) in enumerate(similar_queries):
                prompt_template += f"\nQuery {idx+1}: {similar_q}\nSQL {idx+1}: {similar_sql}\n"
        
        prompt_template += "\n\nInput: {question}\nOutput:"
        
        prompt = PromptTemplate(
            input_variables=["question"],
            template=prompt_template
        )
        
        try:
            sql = self.llm.invoke(prompt.format(question=question)).strip()
            sql = re.sub(r'^```sql\s*|^```\s*|```$', '', sql, flags=re.MULTILINE).strip()
            logger.info(f"Generated SQL: {sql}")
            
            # Cache the successful query
            if sql:
                self.query_cache.add_query(question, sql)
                
            return sql
        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return ""
    
    def validate_sql(self, sql: str):
        dangerous = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'INSERT', 'UPDATE']
        return not any(word in sql.upper() for word in dangerous)

def main():
    st.title("Observability Assistant")
    st.write("Query server metrics in natural language")
    
    db = TelemetryDB()
    rag = RAGPipeline()
    
    # Initialize sample data
    if "data_initialized" not in st.session_state:
        db.insert_sample_data()
        st.session_state["data_initialized"] = True
    
    # Sample queries
    sample_queries = [
        "Show servers with CPU > 80% in last hour",
        "List services with memory > 70% today",
        "Find servers with disk > 85% in last 6 hours"
    ]
    
    st.subheader("Sample Queries:")
    cols = st.columns(len(sample_queries))
    for i, query in enumerate(sample_queries):
        if cols[i].button(query, key=f"sample_{i}"):
            st.session_state['query'] = query
    
    # User input
    query = st.text_input("Your Query:", value=st.session_state.get('query', ''))
    
    if st.button("Run Query") and query:
        with st.spinner("Processing..."):
            sql = rag.generate_sql(query)
            
            if not sql:
                st.error("Failed to generate SQL")
                return
                
            if not rag.validate_sql(sql):
                st.error("Unsafe SQL query detected")
                return
                
            st.subheader("Generated SQL")
            st.code(sql, language="sql")
            
            success, result = db.execute_query(sql)
            if success and not result.empty:
                st.subheader("Results")
                st.dataframe(result)
            elif success:
                st.write("No results found")
            else:
                st.error(f"Error: {result}")

if __name__ == "__main__":
    main()