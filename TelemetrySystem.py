import duckdb
import random
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
import ollama

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelemetryDB:
    """Class to handle database operations for the telemetry system."""
    
    def __init__(self, db_path: str = 'telemetry.duckdb'):
        """Initialize database connection."""
        self.db_path = db_path
        self.conn = duckdb.connect(database=db_path, read_only=False)
        self._init_db()


    
    def _init_db(self) -> None:
        """Initialize database tables if they don't exist."""
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            timestamp TIMESTAMP,
            server_id TEXT,
            service_name TEXT,
            cpu_usage DOUBLE,
            memory_usage DOUBLE,
            disk_usage DOUBLE
        )
        """)
        logger.info("Database initialized")
    
    def insert_sample_data(self, num_rows: int = 500) -> None:
        """Generate and insert sample telemetry data."""
        servers = ["server1", "server2", "server3", "server4", "server5"]
        services = ["auth", "billing", "analytics", "api", "database"]
        now = datetime.now()
        rows = []
        
        for _ in range(num_rows):
            timestamp = now - timedelta(minutes=random.randint(0, 1440)) # last 24h
            server = random.choice(servers)
            service = random.choice(services)
            cpu = round(random.uniform(10, 100), 2)
            memory = round(random.uniform(20, 100), 2)
            disk = round(random.uniform(30, 99.5), 2)
            
           
            if random.random() < 0.1:  
                disk = round(random.uniform(90, 100), 2)
            
            rows.append((timestamp, server, service, cpu, memory, disk))
        
        # Insert into DB
        self.conn.executemany("""
            INSERT INTO metrics (timestamp, server_id, service_name, cpu_usage, memory_usage, disk_usage)
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows)
        logger.info(f"Inserted {num_rows} sample records")
    
    def execute_query(self, query: str) -> Optional[duckdb.DuckDBPyRelation]:
        """Execute a SQL query with proper error handling."""
        try:
            result = self.conn.execute(query)
            return result
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Problematic query: {query}")
            return None
    
    def get_query_result(self, query: str) -> Tuple[bool, Any]:
        """Execute query and return result as DataFrame with success status."""
        result = self.execute_query(query)
        if result is not None:
            try:
                df = result.fetchdf()
                return True, df
            except Exception as e:
                logger.error(f"Error fetching results: {e}")
                return False, str(e)
        return False, "Query execution failed"
    
    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
        logger.info("Database connection closed")


class SQLGenerator:
    """Class to handle LLM-based SQL generation from natural language."""
    
    def __init__(self, model: str = 'mistral'):
        """Initialize with specified LLM model."""
        self.model = model
        self.system_prompt = """
        You are a SQL expert. Convert natural language into SQL queries.
        Assume the telemetry data is in a table named `metrics` with columns:
        (timestamp TIMESTAMP, server_id TEXT, service_name TEXT, cpu_usage DOUBLE, memory_usage DOUBLE, disk_usage DOUBLE).
        
        Rules:
        1. Use DuckDB-compatible syntax
        2. For time-based queries, use INTERVAL syntax like 'now() - INTERVAL 24 HOUR'
        3. Format your output as a clean SQL query only, with no markdown or explanations
        4. Include column names in SELECT statements rather than using *
        5. Use aliases for readability when appropriate
        6. Use proper datetime functions compatible with DuckDB
        
        Return only the SQL query, nothing else.
        """
    
    def generate_sql(self, user_query: str) -> str:
        """Generate SQL from natural language using LLM."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_query}
                ]
            )
            raw_sql = response['message']['content'].strip()
            
            # Clean up the SQL response - remove markdown code blocks if present
            sql = re.sub(r'^```sql\s*|^```\s*|```$', '', raw_sql, flags=re.MULTILINE).strip()
            
            logger.info(f"Generated SQL: {sql}")
            return sql
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return ""
    
    def validate_sql(self, sql: str) -> bool:
        """Basic validation of SQL query to prevent injection and syntax errors."""
        
        dangerous_patterns = [
            r'DROP\s+TABLE',
            r'DELETE\s+FROM',
            r'TRUNCATE\s+TABLE',
            r'ALTER\s+TABLE',
            r'INSERT\s+INTO',
            r'UPDATE\s+.*\s+SET',
            r'EXEC\s+',
          
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                logger.warning(f"Potentially unsafe SQL detected: {pattern}")
                return False
        return True


class TelemetrySystem:
    """Main class orchestrating the telemetry query system."""
    
    def __init__(self, db_path: str = 'telemetry.duckdb', model: str = 'mistral'):
        """Initialize telemetry system with database and LLM components."""
        self.db = TelemetryDB(db_path)
        self.sql_generator = SQLGenerator(model)
    
    def setup_sample_data(self, num_rows: int = 500) -> None:
        """Set up sample data in the database."""
        self.db.insert_sample_data(num_rows)
    
    def process_natural_language_query(self, query: str) -> Tuple[bool, Any, str]:
        """Process a natural language query and return results."""
        # Generate SQL from natural language
        generated_sql = self.sql_generator.generate_sql(query)
        
        if not generated_sql:
            return False, "Failed to generate SQL", ""
        
        # Validate SQL for safety
        if not self.sql_generator.validate_sql(generated_sql):
            return False, "Generated SQL failed safety validation", generated_sql
        
        # Execute the query
        success, result = self.db.get_query_result(generated_sql)
        return success, result, generated_sql
    
    def close(self) -> None:
        """Clean up resources."""
        self.db.close()


def main():
    """Main entry point of the application."""
    # Initialize the system
    telemetry_system = TelemetrySystem()
    
    # Check if we need to populate with sample data
    sample_query = "SELECT COUNT(*) FROM metrics"
    success, result = telemetry_system.db.get_query_result(sample_query)
    
    if success and result.iloc[0, 0] == 0:
        print("Database is empty. Generating sample data...")
        telemetry_system.setup_sample_data(500)
    
    # Example query
    user_query = "List hosts with >90% disk usage in the past 24 hours"
    print(f"\nProcessing query: '{user_query}'")
    
    success, result, generated_sql = telemetry_system.process_natural_language_query(user_query)
    
    print("\nGenerated SQL:")
    print(generated_sql)
    
    if success:
        print("\nQuery Result:")
        print(result)
    else:
        print(f"\nError: {result}")
    
    # Clean up
    telemetry_system.close()


if __name__ == "__main__":
    main()