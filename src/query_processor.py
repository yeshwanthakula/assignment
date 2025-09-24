"""Query processing and intent classification for RAG system."""

import logging
from typing import Tuple
from langchain.prompts import PromptTemplate
from models import QueryIntent, SearchType, QueryResult

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Handles query processing, intent classification, and response generation."""
    
    def __init__(self, llm, graph_manager, vector_manager):
        self.llm = llm
        self.graph_manager = graph_manager
        self.vector_manager = vector_manager
    
    def classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent for routing."""
        query_lower = query.lower()
        
        # Customer service scenarios - route to vector
        customer_service_indicators = [
            'customer asks', 'customer wants', 'customer exceeded',
            'provide options', 'outline next steps', 'advise',
            'what should', 'how to handle', 'recommend solution',"can I"
        ]
        
        # Vector query indicators
        vector_indicators = [
            'similar', 'like', 'about', 'find', 'search', 'issues', 
            'problems', 'recommend', 'suggest', 'best for', 'good for',
            'Describe', 'Explain', 'what is', 'tell me about',
            'describe', 'explain', 'tell me about',"discount"
        ] + customer_service_indicators
        
        # Graph query indicators - only pure structural queries
        graph_indicators = [
            'how many', 'count', 'list all', 'which plans support',
            'what customers are in', 'plans in region',
        ]
        
        # Check customer service patterns first
        if any(indicator in query_lower for indicator in customer_service_indicators):
            return QueryIntent.VECTOR
        elif any(indicator in query_lower for indicator in vector_indicators):
            return QueryIntent.VECTOR
        elif any(indicator in query_lower for indicator in graph_indicators):
            return QueryIntent.GRAPH
        else:
            return QueryIntent.VECTOR  # Default to vector for ambiguous cases
        
    def _determine_search_type(self, query: str) -> SearchType:
        """Determine vector search type based on query content."""
        query_lower = query.lower()
        
        transcript_indicators = [
            'issue', 'problem', 'support', 'transcript', 
            'customer interaction', 'whatsapp'
        ]
        
        if any(indicator in query_lower for indicator in transcript_indicators):
            return SearchType.TRANSCRIPTS
        else:
            return SearchType.PLANS
    
    def process_query(self, query: str) -> QueryResult:
        """Process query and return result with citations."""
        intent = self.classify_intent(query)
        logger.info(f"Query intent: {intent.value} - Question: {query}")
        
        try:
            if intent == QueryIntent.GRAPH:
                answer = self._process_graph_query(query)
                # Extract citations from graph response
                citations = self._extract_citations(answer)
            else:
                answer, citations = self._process_vector_query(query)
            
            return QueryResult(
                answer=answer,
                citations=citations,
                intent=intent
            )
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return QueryResult(
                answer=f"I encountered an error processing your question. Please try rephrasing it.",
                citations=[],
                intent=intent
            )
    
    def _process_graph_query(self, query: str) -> str:
        """Process query using graph database."""
        try:
            return self.graph_manager.query(query)
        except Exception as e:
            logger.warning(f"Graph query failed: {e}")
            raise
    
    def _process_vector_query(self, query: str) -> Tuple[str, list[str]]:
        """Process query using vector search."""
        search_type = self._determine_search_type(query)
        context, citations = self.vector_manager.search(query, search_type)
        
        # Generate answer using LLM
        prompt = self._create_vector_prompt(query, context)
        response = self.llm.invoke(prompt)
        
        return response.content, citations
    
    def _create_vector_prompt(self, query: str, context: str) -> str:
        """Create prompt for vector search response generation."""
        return f"""You are TokuTel's AI assistant. Answer using ONLY the provided context.  You should answer queries about TokuTel's plans and customer support.

RULES:
1. Strictly Use ONLY information from the context provided.
2. Be helpful and conversational while staying factual.
3. If context doesn't contain the answer, say "I don't have that information
4.Do not make up answers or use any information outside the context provided.
"

Context:
{context}

Question: {query}

Answer:"""
    
    def _extract_citations(self, text: str) -> list[str]:
        """Extract citation references from text."""
        import re
        citation_pattern = r'\[([^\]]+\.(?:csv|json|yaml)#[^\]]+)\]'
        matches = re.findall(citation_pattern, text)
        return [f"[{match}]" for match in matches]