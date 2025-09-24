# Hybrid RAG System

A sophisticated hybrid retrieval-augmented generation system combining knowledge graphs and vector search for enterprise applications.

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python -m src.main
   ```

## Important Notes

### Guardrails Setup
⚠️ **Note:** Setting up Guardrails involves some complexity (downloading the model with token). Please refer to this URL for detailed setup instructions: https://www.guardrailsai.com/docs/examples/check_for_pii

**However, the code will run despite the guardrailing not being fully configured.** The system is designed to gracefully handle missing Guardrails dependencies.

### Environment Variables

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Fill in your API keys in the `.env` file:**

**Required API Keys:**
- **Gemini API**: Get free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Neo4j**: Use [Neo4j Aura](https://neo4j.com/aura/) free tier or local installation
- **HuggingFace Token**: Get from [HuggingFace Settings](https://huggingface.co/settings/tokens) (optional, for some models)

**Free Tiers Available:**
- Google Gemini: Free tier with rate limits
- Neo4j Aura: Free tier with 200k nodes + 400k relationships
- HuggingFace: Free tier for most open models

## Project Structure
```
├── src/
│   ├── main.py           # Main application entry point
│   ├── hybrid_rag.py     # Core RAG implementation
│   ├── vector_manager.py # Vector search functionality
│   ├── graph_manager.py  # Knowledge graph operations
│   └── ...
├── data/                 # Data files
├── images/              # Architecture diagrams
├── requirements.txt     # Python dependencies
└── architecture.md      # Detailed system architecture
```

## Example Results & Training Code

- **📋 Sample Outputs**: You can find the results for various prompts in `example_logs.txt`
- **🔬 Training Code**: The training code for spaCy model is available in the `notebooks/` folder

## Next Steps
After successful installation, check the `architecture.md` file for detailed information about the system design and components.

## Troubleshooting
- If you encounter dependency issues, ensure you're using Python 3.8+
- For Neo4j connection issues, verify your database credentials
- For Guardrails setup, refer to their official documentation linked above