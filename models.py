"""Data models and types for TokuTel RAG system."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum

class QueryIntent(Enum):
    """Query intent classification."""
    GRAPH = "graph"
    VECTOR = "vector"

class SearchType(Enum):
    """Vector search types."""
    PLANS = "plans"
    TRANSCRIPTS = "transcripts"
    KB = "kb"

@dataclass
class CitationSource:
    """Track citation sources with exact locations."""
    filename: str
    location: str
    content: Dict[str, Any]
    node_type: Optional[str] = None

@dataclass
class QueryResult:
    """Result of a query with citations."""
    answer: str
    citations: List[str]
    intent: QueryIntent
    
@dataclass 
class PolicyCheck:
    """Policy enforcement check result."""
    passed: bool
    violations: List[str]
    escalation_level: Optional[str] = None