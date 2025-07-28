from langgraph.graph import StateGraph
from typing import Dict, Any, TypedDict
import structlog
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from privacy.masking import DatabaseMasker, DocumentMasker
from db.vector_store import FAISSVectorStore

logger = structlog.get_logger()

class AgentState(TypedDict):
    query: str
    masked_query: str
    session_id: str
    db_session: Any
    pii_mapping: Dict
    context: list
    sql_query: str
    final_answer: str

class AgentGraph:
    def __init__(
        self,
        db_masker: DatabaseMasker,
        doc_masker: DocumentMasker,
        vector_store: FAISSVectorStore,
        model_name: str = "microsoft/phi-3-mini"
    ):
        self.db_masker = db_masker
        self.doc_masker = doc_masker
        self.vector_store = vector_store
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def mask_query(self, state: AgentState) -> AgentState:
        """Mask sensitive information in the query"""
        try:
            masked_content, mapping = self.doc_masker.mask_document(
                state["query"].encode()
            )
            state["masked_query"] = masked_content
            state["pii_mapping"] = mapping
            return state
        except Exception as e:
            logger.error("Query masking error", error=str(e))
            raise
    
    def retrieve_context(self, state: AgentState) -> AgentState:
        """Retrieve relevant context from vector store"""
        try:
            results = self.vector_store.search(state["masked_query"])
            state["context"] = results
            return state
        except Exception as e:
            logger.error("Context retrieval error", error=str(e))
            raise
    
    def generate_response(self, state: AgentState) -> AgentState:
        """Generate response using Phi-3-mini"""
        try:
            # Prepare prompt with context
            prompt = self._prepare_prompt(state)
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Unmask the response
            unmasked_response = self.doc_masker.unmask_document(
                response,
                state["pii_mapping"]
            )
            
            state["final_answer"] = unmasked_response
            return state
        except Exception as e:
            logger.error("Response generation error", error=str(e))
            raise
    
    def _prepare_prompt(self, state: AgentState) -> str:
        """Prepare prompt with context"""
        context_str = "\n".join([f"Context {i+1}: {doc}" for i, (doc, _) in enumerate(state["context"])])
        
        prompt = f"""
Based on the following context and query, provide a comprehensive answer.

Context:
{context_str}

Query: {state["masked_query"]}

Answer:"""
        return prompt

def build_agent_graph(
    db_masker: DatabaseMasker,
    doc_masker: DocumentMasker,
    vector_store: FAISSVectorStore
) -> StateGraph:
    """Build the agent workflow graph"""
    agent = AgentGraph(db_masker, doc_masker, vector_store)
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("mask_query", agent.mask_query)
    workflow.add_node("retrieve_context", agent.retrieve_context)
    workflow.add_node("generate_response", agent.generate_response)
    
    # Add edges
    workflow.add_edge("mask_query", "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_response")
    
    workflow.set_entry_point("mask_query")
    
    return workflow
