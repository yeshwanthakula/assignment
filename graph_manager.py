"""Neo4j graph database management for TokuTel RAG system."""

import logging
from typing import Dict, Any, List
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain.prompts import PromptTemplate

from config import Neo4jConfig

logger = logging.getLogger(__name__)

class GraphManager:
    """Manages Neo4j knowledge graph operations."""
    
    def __init__(self, neo4j_config: Neo4jConfig, llm):
        self.config = neo4j_config
        self.llm = llm
        self.neo4j_graph = None
        self.cypher_chain = None
        
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test Neo4j connection."""
        try:
            with GraphDatabase.driver(self.config.uri, auth=(self.config.username, self.config.password)) as driver:
                driver.verify_connectivity()
                logger.info("Neo4j connection established")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise
    
    def build_knowledge_graph(self, data: Dict[str, Any]) -> None:
        """Build knowledge graph from loaded data."""
        logger.info("Building knowledge graph...")
        
        with GraphDatabase.driver(self.config.uri, auth=(self.config.username, self.config.password)) as driver:
            with driver.session(database=self.config.database) as session:
                self._clear_graph(session)
                self._create_company_node(session, data['kb'])
                self._create_plan_nodes(session, data['plans'])
                self._create_feature_nodes(session, data['kb'])
                self._create_customer_nodes(session, data['transcripts'])
        
        logger.info("Knowledge graph built successfully")
    
    def _clear_graph(self, session) -> None:
        """Clear existing graph data."""
        session.run("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared")
    
    def _create_company_node(self, session, kb_data: Dict[str, Any]) -> None:
        """Create company node."""
        session.run("""
            CREATE (c:Company {
                name: $name,
                regions: $regions,
                citation: '[kb.yaml#company]'
            })
        """, 
        name=kb_data['company']['name'],
        regions=kb_data['company']['region_focus'])
    
    def _create_plan_nodes(self, session, plans_df) -> None:
        """Create plan nodes with proper citations."""
        for idx, plan in plans_df.iterrows():
            citation = f"[plans.csv#row={idx+2}]"
            session.run("""
                CREATE (p:Plan {
                    plan_id: $plan_id,
                    name: $name,
                    region: $region,
                    minutes: $minutes,
                    sms: $sms,
                    price_usd: $price_usd,
                    notes: $notes,
                    citation: $citation
                })
            """, 
            plan_id=plan['plan_id'],
            name=plan['name'],
            region=plan['region'],
            minutes=int(plan['minutes']),
            sms=int(plan['sms']),
            price_usd=float(plan['price_usd']),
            notes=plan['notes'],
            citation=citation)
    
    def _create_feature_nodes(self, session, kb_data: Dict[str, Any]) -> None:
        """Create feature nodes and relationships."""
        features = kb_data['features_matrix']
        
        for feature_name, supported_plans in features.items():
            # Create feature node
            session.run("""
                CREATE (f:Feature {
                    name: $name,
                    citation: '[kb.yaml#features_matrix]'
                })
            """, name=feature_name)
            
            # Connect to supporting plans
            for plan_id in supported_plans:
                session.run("""
                    MATCH (p:Plan {plan_id: $plan_id})
                    MATCH (f:Feature {name: $feature_name})
                    CREATE (p)-[:SUPPORTS]->(f)
                """, plan_id=plan_id, feature_name=feature_name)
    
    def _create_customer_nodes(self, session, transcripts: List[Dict[str, Any]]) -> None:
        """Create customer and transcript nodes."""
        for transcript in transcripts:
            # Create customer
            session.run("""
                MERGE (c:Customer {
                    name: $customer_name,
                    region: $region,
                    citation: $citation
                })
            """, 
            customer_name=transcript['customer'],
            region=transcript['region'],
            citation=f"[transcripts.json#{transcript['id']}]")
            
            # Create transcript
            session.run("""
                MATCH (c:Customer {name: $customer_name})
                CREATE (t:Transcript {
                    id: $id,
                    channel: $channel,
                    text: $text,
                    citation: $citation
                })
                CREATE (c)-[:HAS_INTERACTION]->(t)
            """, 
            customer_name=transcript['customer'],
            id=transcript['id'],
            channel=transcript['channel'],
            text=transcript['text'],
            citation=f"[transcripts.json#{transcript['id']}]")
    
    def setup_cypher_chain(self) -> bool:
        """Setup GraphCypherQAChain for natural language queries."""
        try:
            self.neo4j_graph = Neo4jGraph(
                url=self.config.uri,
                username=self.config.username,
                password=self.config.password,
                database=self.config.database,
                refresh_schema=True,
                sanitize=True
            )
            
            self.neo4j_graph.refresh_schema()
            logger.info("Neo4j Graph schema refreshed")
            
            # Custom prompts for better results
            cypher_prompt = PromptTemplate(
                input_variables=["schema", "question"], 
                template="""
You are a Neo4j Cypher expert.  Understand the question thoroughly and the Generate a Cypher query to answer the question.

IMPORTANT RULES:
1. Feature names are lowercase with underscores: call_recording, sentiment_analysis, whatsapp_api, sso
2. Regions are 2-letter codes: SG, MY, ID, TH, PH (not full country names)
3. For "how many plans" queries, use MATCH (p:Plan) to get ALL plans
4. Use OPTIONAL MATCH for features to include plans without features
5. Always include 'citation' property in RETURN statements
6. Generate ONLY the Cypher query

Graph Schema:
{schema}

Question: {question}

Examples:
- "call recording" → use f.name = "call_recording"  
- "Singapore" → use c.region = "SG"
- "WhatsApp" → use f.name = "whatsapp_api"

Cypher query:"""
            )
            
            answer_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
You are TokuTel's AI assistant. Answer using ONLY the provided query results.

RULES:
1. Use ONLY information from the query results
2. Include all citations as [filename#location]
3. Be conversational while staying factual
4. If no results, say "No information found"

Query Results: {context}
Question: {question}

Answer with citations:"""
            )
            
            self.cypher_chain = GraphCypherQAChain.from_llm(
                llm=self.llm,
                graph=self.neo4j_graph,
                verbose=True,
                cypher_prompt=cypher_prompt,
                qa_prompt=answer_prompt,
                validate_cypher=True,
                return_intermediate_steps=True,
                allow_dangerous_requests=True
            )
            
            logger.info("GraphCypherQAChain successfully created")
            return True
            
        except Exception as e:
            logger.error(f"GraphCypherQAChain setup failed: {e}")
            return False
    
    def query(self, question: str) -> str:
        """Execute graph query using Cypher chain."""
        if not self.cypher_chain:
            raise RuntimeError("Cypher chain not initialized")
        
        try:
            result = self.cypher_chain.invoke({"query": question})
            return result["result"]
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            raise