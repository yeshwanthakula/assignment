"""Data loading utilities for RAG system."""

import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging

from models import CitationSource

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and parsing of data files."""
    
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.citation_sources = {}
    
    def load_all_data(self) -> Dict[str, Any]:
        """Load all data files and return as dictionary."""
        data = {}
        
        # Load plans
        data['plans'] = self._load_plans()
        logger.info(f"Loaded {len(data['plans'])} plans")
        
        # Load transcripts
        data['transcripts'] = self._load_transcripts()
        logger.info(f"Loaded {len(data['transcripts'])} transcripts")
        
        # Load knowledge base
        data['kb'] = self._load_knowledge_base()
        logger.info("Loaded knowledge base")
        
        # Register citations
        self._register_citations(data)
        logger.info(f"Registered {len(self.citation_sources)} citation sources")
        
        return data
    
    def _load_plans(self) -> pd.DataFrame:
        """Load plans CSV file."""
        if not Path(self.data_paths.plans_csv).exists():
            raise FileNotFoundError(f"Plans file not found: {self.data_paths.plans_csv}")
        return pd.read_csv(self.data_paths.plans_csv)
    
    def _load_transcripts(self) -> List[Dict[str, Any]]:
        """Load transcripts JSON file."""
        if not Path(self.data_paths.transcripts_json).exists():
            raise FileNotFoundError(f"Transcripts file not found: {self.data_paths.transcripts_json}")
        
        with open(self.data_paths.transcripts_json, 'r') as f:
            return json.load(f)
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load knowledge base YAML file."""
        if not Path(self.data_paths.kb_yaml).exists():
            raise FileNotFoundError(f"KB file not found: {self.data_paths.kb_yaml}")
        
        with open(self.data_paths.kb_yaml, 'r') as f:
            return yaml.safe_load(f)
    
    def _register_citations(self, data: Dict[str, Any]) -> None:
        """Register all data sources for citation tracking."""
        
        # Register plans data
        for idx, plan in data['plans'].iterrows():
            citation_id = f"plan_{plan['plan_id']}"
            self.citation_sources[citation_id] = CitationSource(
                filename="plans.csv",
                location=f"row={idx+2}",
                content=plan.to_dict(),
                node_type="Plan"
            )
        
        # Register transcript data  
        for transcript in data['transcripts']:
            citation_id = f"transcript_{transcript['id']}"
            self.citation_sources[citation_id] = CitationSource(
                filename="transcripts.json",
                location=transcript['id'],
                content=transcript,
                node_type="Transcript"
            )
        
        # Register KB sections
        kb_sections = ['company', 'pricing_rules', 'features_matrix']
        for section in kb_sections:
            citation_id = f"kb_{section}"
            self.citation_sources[citation_id] = CitationSource(
                filename="kb.yaml",
                location=section,
                content=data['kb'].get(section, {}),
                node_type="Policy"
            )