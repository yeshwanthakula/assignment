"""Vector store management for semantic search."""

import logging
import os
from typing import Dict, Any, List, Tuple
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from models import SearchType
from config import EmbeddingConfig

logger = logging.getLogger(__name__)

class VectorManager:
    """Manages vector stores for semantic search."""
    
    def __init__(self, embedding_config: EmbeddingConfig, hf_token: str):
        self.config = embedding_config
        self.embeddings = self._setup_embeddings(hf_token)
        self.plan_vectorstore = None
        self.transcript_vectorstore = None
    
    def _setup_embeddings(self, hf_token: str) -> HuggingFaceEmbeddings:
        """Initialize HuggingFace embeddings."""
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        
        return HuggingFaceEmbeddings(
            model_name=self.config.model_name,
            model_kwargs={'device': self.config.device},
            encode_kwargs={'normalize_embeddings': self.config.normalize_embeddings}
        )
    
    def build_vectorstores(self, data: Dict[str, Any]) -> None:
        """Build vector stores from loaded data."""
        logger.info("Building vector stores...")
        
        # Create plan documents
        plan_documents = self._create_plan_documents(data['plans'])
        
        # Create transcript documents  
        transcript_documents = self._create_transcript_documents(data['transcripts'])
        
        # Create KB policy documents
        kb_documents = self._create_kb_documents(data['kb'])
        
        # Combine all documents for unified search
        all_documents = plan_documents + transcript_documents + kb_documents
        
        # Create unified vector store
        self.unified_vectorstore = FAISS.from_documents(all_documents, self.embeddings)
        
        # Keep separate stores for targeted search
        self.plan_vectorstore = FAISS.from_documents(plan_documents, self.embeddings)
        self.transcript_vectorstore = FAISS.from_documents(transcript_documents, self.embeddings)
        
        logger.info(f"Vector stores created: {len(all_documents)} total documents")

    def _create_kb_documents(self, kb_data: Dict[str, Any]) -> List[Document]:
        """Create documents from knowledge base policies."""
        documents = []
        if 'sla' in kb_data['company']:
            sla = kb_data['company']['sla']
            print("Adding SLA")
            content = f"SLA Response Times:\nStandard: {sla['standard_response_hours']} hours\nEnterprise: {sla['enterprise_response_hours']} hours"
            documents.append(Document(
                page_content=content,
                metadata={'citation': '[kb.yaml#sla]', 'type': 'policy'}
            ))
    
    # Add escalation policies
        if 'escalation_policy' in kb_data['company']:
            escalation = kb_data['company']['escalation_policy']
            content = "Escalation Policies:\n"
            print("Adding escalation policies")
            for level, desc in escalation.items():
                content += f"{level.upper()}: {desc}\n"
            documents.append(Document(
                page_content=content,
                metadata={'citation': '[kb.yaml#escalation_policy]', 'type': 'policy'}
            ))
        
    
    # Add PII policy
        if 'pii_policy' in kb_data.get('company', {}):
            pii = kb_data['company']['pii_policy']
            content = "PII Policy:\n" + "\n".join([f"- {rule}" for rule in pii])
            documents.append(Document(
                page_content=content,
                metadata={'citation': '[kb.yaml#company]', 'type': 'policy'}
            ))
    
        
        # Pricing rules
        if 'pricing_rules' in kb_data:
            pricing = kb_data['pricing_rules']
            content = f"Pricing Rules:\n"
            if 'discounts' in pricing:
                content += "Available discounts:\n"
                for discount in pricing['discounts']:
                    content += f"- {discount['condition']}: {discount['value_pct']}% discount\n"
            
            doc = Document(
                page_content=content,
                metadata={
                    'citation': '[kb.yaml#pricing_rules]',
                    'type': 'policy',
                    'section': 'pricing_rules'
                }
            )
            documents.append(doc)
        
        return documents
    
    def _create_plan_documents(self, plans_df) -> List[Document]:
        """Create LangChain documents from plans data."""
        documents = []
        
        for idx, plan in plans_df.iterrows():
            doc = Document(
            page_content=f"Plan ID: {plan['plan_id']}\nFull Name: {plan['name']}\nDescription: {plan['notes']}\nPrice: ${plan['price_usd']}\nRegion: {plan['region']}\nMinutes: {plan['minutes']}\nSMS: {plan['sms']}",
            metadata={
                'plan_id': plan['plan_id'],
                'name': plan['name'],  # Add full name
                'citation': f"[plans.csv#row={idx+2}]",
                'type': 'plan'
            }
        )
            documents.append(doc)
        
        return documents
    
    def _create_transcript_documents(self, transcripts: List[Dict[str, Any]]) -> List[Document]:
        """Create LangChain documents from transcripts data."""
        documents = []
        
        for transcript in transcripts:
            doc = Document(
                page_content=f"Customer: {transcript['customer']}\nChannel: {transcript['channel']}\nInteraction: {transcript['text']}",
                metadata={
                    'transcript_id': transcript['id'],
                    'customer': transcript['customer'],
                    'citation': f"[transcripts.json#{transcript['id']}]",
                    'type': 'transcript',
                    'channel': transcript['channel'],
                    'region': transcript['region']
                }
            )
            documents.append(doc)
        
        return documents
    
    def search(self, query: str, search_type: SearchType, k: int = 3) -> Tuple[str, List[str]]:
        """Perform semantic search and return context with citations."""
        # if any(word in query.lower() for word in ['discount', 'pricing', 'policy', 'rule', 'nonprofit', 'annual']):
        #     vectorstore = self.unified_vectorstore if hasattr(self, 'unified_vectorstore') else self.plan_vectorstore
        # else:
        #     vectorstore = (self.plan_vectorstore if search_type == SearchType.PLANS 
        #               else self.transcript_vectorstore)
        vectorstore = self.unified_vectorstore
        import numpy as np
        np.random.seed(42)
        
        if not vectorstore:
            raise RuntimeError(f"Vector store for {search_type.value} not initialized")
        
        # Get relevant documents
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
        docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1])
        
        # Filter by relevance threshold (lower score = more similar)
        relevant_docs = [(doc, score) for doc, score in docs_with_scores if score < 0.7]
        
        if not relevant_docs:
            # If no docs meet threshold, take the best one
            relevant_docs = [docs_with_scores[0]] if docs_with_scores else []
        
        # Extract content and citations from relevant docs only
        context_parts = []
        citations = []
        
        for doc, score in relevant_docs:
            logger.info(f"Retrieved: {doc.metadata.get('plan_id', 'unknown')} - {doc.metadata['citation']} - Score: {score}")
            context_parts.append(doc.page_content)
            if 'citation' in doc.metadata:
                citations.append(doc.metadata['citation'])
        
        context = "\n\n".join(context_parts)
        return context, citations