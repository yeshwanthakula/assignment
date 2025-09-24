"""TokuTel RAG System - Main orchestrator class."""

import logging
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config
from data_loader import DataLoader
from graph_manager import GraphManager
from .vector_manager import VectorManager
from .query_processor import QueryProcessor
from models import QueryResult

logger = logging.getLogger(__name__)

class TokuTelRAG:
    """Main TokuTel RAG system orchestrator."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config.data_paths)
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.llm.model_name,
            google_api_key=config.llm.gemini_api_key,
            temperature=0,
            max_tokens=config.llm.max_tokens
        )
        logger.info("LLM initialized")
        
        # Initialize managers
        self.graph_manager = GraphManager(config.neo4j, self.llm)
        self.vector_manager = VectorManager(config.embeddings, config.llm.hf_token)
        
        # Query processor will be initialized after data loading
        self.query_processor = None
        
        # Data storage
        self.data = None
    
    def setup(self) -> None:
        """Setup the complete RAG system."""
        logger.info("Setting up TokuTel RAG system...")
        
        # Load data
        self.data = self.data_loader.load_all_data()
        
        # Build knowledge graph
        self.graph_manager.build_knowledge_graph(self.data)
        
        # Build vector stores
        self.vector_manager.build_vectorstores(self.data)
        
        # Setup graph cypher chain
        graph_success = self.graph_manager.setup_cypher_chain()
        
        # Initialize query processor
        self.query_processor = QueryProcessor(
            self.llm,
            self.graph_manager,
            self.vector_manager
        )
        
        logger.info("TokuTel RAG system setup complete")
        
        if graph_success:
            logger.info("Graph queries: GraphCypherQAChain with Gemini")
        else:
            logger.warning("Graph queries: Fallback mode")
    
    def query(self, question: str) -> str:
        """Process a user query and return formatted response."""
        if not self.query_processor:
            raise RuntimeError("System not initialized. Call setup() first.")
        
        result = self.query_processor.process_query(question)
        return self._format_response(result)
    
    def _format_response(self, result: QueryResult) -> str:
        """Format query result with citations."""
        response = result.answer
        
        # Add citations if not already included
        if result.citations and not any(citation in response for citation in result.citations):
            unique_citations = list(set(result.citations))
            response += f"\n\nSources: {', '.join(unique_citations)}"
        
        return response