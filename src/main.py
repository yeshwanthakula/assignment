"""Main entry point for TokuTel RAG system."""

import logging
import sys
from pathlib import Path
from gaurdrail import gaurd_rail_output,mask_phone_numbers,load_model,predict_escalation,escalate
from config import Config
from .hybrid_rag import TokuTelRAG


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_demo_queries(rag_system: TokuTelRAG):
    """Run demonstration queries to showcase system capabilities."""
    demo_queries = [
        # Graph queries (structured)
        "Which plans support call recording?",
        "How many plans are available?", 
        "What customers are in Singapore?",
        
        # Vector queries (semantic)
        "Find plans similar to contact center solutions",
        "Show me issues about WhatsApp problems", 
        "Recommend a plan for unified communications",
        "What plans are good for call centers?",
        "Describe the Kapok Omni Plus plan",
        "Critical data loss reported what to do?",
        "Bulk email delivery failing what to do?",
        "What is the SLA for enterprise hours?",
        "how many plans are there and what do they offer",

        "Customer asks to enable call recording on CC Lite. Provide options",
        "Customer exceeded WhatsApp quota; outline next steps.",
        "Customer wants SSO support; advise plan choice.",
        "Customer asks about nonprofit discount and annual prepay together."
        "Can I get an annual discount?"

    ]
    
    print("\n" + "="*60)
    print("TESTING TOKUTEL RAG SYSTEM")
    print("="*60)
    spacy_esclation_model = load_model()

    from time import sleep
    sleep(2) # for rate limit issues
    
    for query in demo_queries:
        print(f"\nQuestion: {query}")
        print("-" * 50)
        try:
            level = predict_escalation(spacy_esclation_model,query)
            if level:
                escalate(level)
                continue
        except Exception as e:
            print("error in escaltation.. skipping")
            pass
        
        try:
            response = rag_system.query(query)
            try:
                original,masked = gaurd_rail_output(response)
                response = mask_phone_numbers(original , masked)
                print(f"Agent:\n{response}")
            except Exception as e:
                print(f"Agent(without gaurdrailing):\n{response}")
        except Exception as e:
            logger.error(f"Query failed: {e}")
            print(f"Error: {e}")
        
        print()

def run_interactive_mode(rag_system: TokuTelRAG):
    """Run interactive query mode."""
    print("\nInteractive Mode (type 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\nYour question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input:
                response = rag_system.query(user_input)
                print(f"\n{response}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Error: {e}")

def main():
    """Main entry point."""
    try:
        # Load configuration
        config = Config()
        logger.info("Configuration loaded")
        
        # Initialize RAG system
        rag_system = TokuTelRAG(config)
        
        # Setup system
        print("Setting up TokuTel RAG system...")
        rag_system.setup()
        print("Setup complete!")
        
        # Check command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "--demo":
                run_demo_queries(rag_system)
                return
            elif sys.argv[1] == "--interactive":
                run_interactive_mode(rag_system)
                return
        
        # Default: run both demo and interactive
        run_demo_queries(rag_system)
        run_interactive_mode(rag_system)
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()