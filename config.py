"""Configuration settings for TokuTel RAG system."""

import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

load_dotenv()

dbname = os.getenv("NEO4J_DATABASE")

@dataclass
class Neo4jConfig:
    """Neo4j database configuration."""
    uri: str
    username: str
    password: str
    database: str = dbname

@dataclass
class LLMConfig:
    """Language model configuration."""
    gemini_api_key: str
    hf_token: str
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.1
    max_tokens: int = 1024

@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: str = "cpu"
    normalize_embeddings: bool = True

@dataclass
class DataPaths:
    """Data file paths."""
    plans_csv: str = "data/plans.csv"
    transcripts_json: str = "data/transcripts.json" 
    kb_yaml: str = "data/kb.yaml"
    faq_jsonl: str = "data/faq.jsonl"
    eval_prompts: str = "eval_prompts.txt"

class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.neo4j = Neo4jConfig(
            uri=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        
        self.llm = LLMConfig(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            hf_token=os.getenv("HUGGINGFACE_TOKEN")
        )
        
        self.embeddings = EmbeddingConfig()
        self.data_paths = DataPaths()
        
        # Validate required environment variables
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required configuration is present."""
        if not self.llm.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        if not self.llm.hf_token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable is required")