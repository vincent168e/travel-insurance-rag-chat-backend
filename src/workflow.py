from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

from src.config import settings
from src.database import query_vector_db
from src.guardrails import check_guardrails

logging.basicConfig(level=logging.INFO)

# State definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    final_response: str

# Initialize Model
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=settings.GEMINI_API_KEY)

# --- HELPER FUNCTION FOR STRIP/LIST SAFETY ---
def extract_text_content(content) -> str:
    """Safely extracts a string from LangChain message contents regardless of SDK version variations."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        extracted = []
        for block in content:
            if isinstance(block, str):
                extracted.append(block)
            elif isinstance(block, dict) and "text" in block:
                extracted.append(block["text"])
        return "".join(extracted)
    return str(content)

# --- NODES ---

def guardrail_node(state: AgentState):
    last_message = state["messages"][-1].content
    if check_guardrails(last_message):
        return {"final_response": "please ask questions related to travel insurance policy"}
    return {"final_response": ""}

def retrieve_and_scope_node(state: AgentState):
    # Short circuit if guardrails failed
    if state.get("final_response"):
        return state

    last_message = state["messages"][-1].content
    
    # 1. Vector DB Retrieval
    docs, max_score = query_vector_db(last_message)
    
    # Check similarity threshold (Example: 0.40 cutoff)
    if max_score < 0.40:
        return {"final_response": "sorry, no related information can be provided"}
        
    context_str = "\n\n".join(docs)
    
    # 2. LLM Out-of-Scope Classification
    scope_prompt = (
        f"Analyze this user question: '{last_message}'. "
        "Is this question related to travel insurance policies? "
        "Reply strictly with 'YES' or 'NO'."
    )

    raw_response = llm.invoke([HumanMessage(content=scope_prompt)])

    scope_check = extract_text_content(raw_response.content).strip().upper()
    
    if "NO" in scope_check:
        return {"final_response": "That's a bit outside what I can help with. Could we focus on your travel insurance policy instead?"}
        
    return {"context": context_str}

def generate_node(state: AgentState):
    if state.get("final_response"):
        return state
        
    last_message = state["messages"][-1].content
    context = state["context"]
    
    system_prompt = (
        "You are a professional, warm, and friendly customer service agent for Blue Cross Travel Insurance.\n"
        "Your tone should be welcoming, empathetic, polite, and reassuring, ensuring the customer feels supported.\n\n"
        "CRITICAL RULES:\n"
        "1. Answer the user's question accurately using ONLY the provided policy context below.\n"
        "2. Do not use outside knowledge or extrapolate. Keep the information factually tied to the context.\n"
        "3. If you cannot answer using the context, disregard your warm persona rules and state exactly: "
        "'I don't have that info right now, but I'm here to help with anything else.'\n\n"
        f"Context:\n{context}"
    )
    
    # Combine conversation history with the system rule
    messages = [HumanMessage(content=system_prompt)] + list(state["messages"])
    raw_response = llm.invoke(messages)
    
    final_text = extract_text_content(raw_response.content)
    
    return {"final_response": final_text}

# --- EDGE ROUTING ROUTINES ---

def route_after_guardrail(state: AgentState):
    if state.get("final_response"):
        return "end"
    return "continue"

def route_after_eval(state: AgentState):
    if state.get("final_response"):
        return "end"
    return "continue"

# --- GRAPH COMPOSER ---

workflow = StateGraph(AgentState)

workflow.add_node("guardrail", guardrail_node)
workflow.add_node("retrieve_and_scope", retrieve_and_scope_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "guardrail")

workflow.add_conditional_edges(
    "guardrail",
    route_after_guardrail,
    {"end": END, "continue": "retrieve_and_scope"}
)

workflow.add_conditional_edges(
    "retrieve_and_scope",
    route_after_eval,
    {"end": END, "continue": "generate"}
)

workflow.add_edge("generate", END)

# In-Memory session tracking checkpointer
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

# ==========================================
# 💬 INTERACTIVE CLI TEST RUNNER
# ==========================================
if __name__ == "__main__":
    import uuid
    
    # Generate a persistent session thread for this terminal run
    session_thread = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_thread}}
    
    print("\n🚀 Interactive LangGraph Agent Test Environment")
    print(f"🧵 Session Started (Thread ID: {session_thread})")
    print("👉 Type your prompt and press Enter. Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            # Capture user input
            user_prompt = input("\n👤 You: ")
            
            # Check for exit command
            if user_prompt.strip().lower() in ["exit", "quit"]:
                print("\n👋 Session ended. Goodbye!")
                break
                
            # Skip empty inputs
            if not user_prompt.strip():
                continue
                
            # Run the prompt through the LangGraph state machine
            output = app_graph.invoke(
                {"messages": [HumanMessage(content=user_prompt)]}, 
                config=config
            )
            
            # Print the final result
            print(f"🤖 Agent: {output['final_response']}")
            
        except KeyboardInterrupt:
            print("\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error executing workflow: {e}")