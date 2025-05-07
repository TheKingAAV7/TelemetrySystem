 query telemetry data using natural language, which is converted into SQL using an LLM (via Ollama), and executed on a DuckDB database. A Streamlit frontend makes it interactive and easy to use.

---

## 🚀 Features

- Natural language to SQL conversion using LLM (`mistral` via Ollama)
- Executes queries on DuckDB
- Streamlit-based UI
- Sample telemetry data generation (server metrics)
- SQL safety validation

---

## 🛠️ Installation

1. **Clone the repo**

```bash
git clone https://github.com/your-username/telemetry-nl-query.git
cd telemetry-nl-query

2. **Install dependencies**
pip install -r requirements.txt



3. Run Ollama (if not already running)

Make sure Ollama is installed and the mistral model is pulled:
ollama run mistral



4. Start the Streamlit app

streamlit run app.py
📝 Example Queries
List hosts with >90% disk usage in the past 24 hours

Show average CPU usage per service in the last 12 hours

Which servers had memory usage above 80%?

📁 Project Structure
bash
Copy code
.
├── newapp.py              
├── telemetry.py        
├── telemetry.duckdb    
├── requirements.txt
└── README.md