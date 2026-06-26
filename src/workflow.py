import json
import logging
import re
import sys
from pathlib import Path
from typing import Annotated, Any, Sequence, TypedDict
import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import settings
from src.database import query_policy_chunks
from src.guardrails import check_guardrails, detect_emergency
from src.messages import MESSAGES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
	messages: Annotated[Sequence[BaseMessage], add_messages]
	user_input: str
	service_category: str
	claim_category: str
	claim_description: str
	image_urls: list[str]
	ocr_extracts: list[str]
	policy_tier: str
	policy_context: str
	# citations: list[str]
	pending_claim_transition: bool
	pending_claim_scenarios: list[str]
	claim_stage: str | None
	clarification_attempts: int
	audit_report: dict[str, Any]
	next_agent: str
	final_response: str
	session_closed: bool
	escalation_reason: str
	last_activity_timestamp: float


FAST_CLASSIFIER_MODEL = "gemini-3.1-flash-lite"

classifier_llm = ChatGoogleGenerativeAI(
	model=FAST_CLASSIFIER_MODEL,
	google_api_key=settings.GEMINI_API_KEY,
	temperature=0,
)

policy_llm = ChatGoogleGenerativeAI(
	model="gemini-3.1-flash-lite",
	google_api_key=settings.GEMINI_API_KEY,
	temperature=0,
)

action_llm = ChatGoogleGenerativeAI(
	model="gemini-3.1-flash-lite",
	google_api_key=settings.GEMINI_API_KEY,
	temperature=0,
)

validation_llm = ChatGoogleGenerativeAI(
	model="gemini-3.1-flash-lite",
	google_api_key=settings.GEMINI_API_KEY,
	temperature=0,
)


HARD_CODED_POLICY_TIER = "Single-trip solutions Canada package"
HARD_CODED_PREPAYMENT_OPTION = True


def log_event(event: str, **details: Any) -> None:
	"""Emit compact structured logs for workflow routing and agent activity."""
	parts: list[str] = [event]
	for key, value in details.items():
		if isinstance(value, str) and len(value) > 220:
			value = f"{value[:217]}..."
		parts.append(f"{key}={value!r}")
	logger.info(" | ".join(parts))


def log_node_start(node_name: str, state: AgentState) -> None:
	log_event(
		f"{node_name}.start",
		thread_id=state.get("messages", [])[0].additional_kwargs.get("thread_id") if state.get("messages") else None,
		service_category=state.get("service_category"),
		claim_stage=state.get("claim_stage"),
		clarification_attempts=state.get("clarification_attempts", 0),
		final_response_present=bool(state.get("final_response")),
		user_input=state.get("user_input", ""),
	)


def log_node_end(node_name: str, result: dict[str, Any]) -> None:
	log_event(
		f"{node_name}.end",
		next_agent=result.get("next_agent"),
		service_category=result.get("service_category"),
		claim_stage=result.get("claim_stage"),
		session_closed=bool(result.get("session_closed", False)),
		final_response=result.get("final_response", ""),
	)


def extract_text_content(content: Any) -> str:
	"""Safely extracts a plain string from SDK-dependent response payload formats."""
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		extracted: list[str] = []
		for block in content:
			if isinstance(block, str):
				extracted.append(block)
			elif isinstance(block, dict) and "text" in block:
				extracted.append(str(block["text"]))
		return "".join(extracted)
	return str(content)


def normalize_user_text(message_like: Any) -> str:
	if isinstance(message_like, str):
		return message_like
	return extract_text_content(message_like)


def decompose_complex_query(text: str) -> dict[str, Any]:
    """Uses the fast classifier to safely unpack complex, multi-part inquiries."""
    prompt = (
        "You are an AI routing assistant for a travel insurance platform.\n"
        "Analyze the user's input and determine if they are asking a policy question, "
        "trying to file claims for multiple incidents, or doing both (mixed).\n\n"
        "Decompose the input into a clean policy inquiry and an explicit list of claim scenarios.\n\n"
        "Return ONLY a JSON object with this exact schema:\n"
        "{\n"
        "  \"intent\": \"inquiry\" | \"claim\" | \"mixed\",\n"
        "  \"policy_inquiry\": \"The isolated coverage question text, or null if none\",\n"
        "  \"claim_scenarios\": [\"List of isolated descriptions for each claim incident mentioned\"]\n"
        "}\n\n"
        f"User Input: {text}\n"
    )
    try:
        raw = classifier_llm.invoke([HumanMessage(content=prompt)])
        return parse_json_object(extract_text_content(raw.content))
    except Exception as exc:
        logger.warning("LLM query decomposition failed, falling back to basic routing: %s", exc)
        return {"intent": "inquiry", "policy_inquiry": text, "claim_scenarios": []}


# def format_citations(chunks: list[dict[str, Any]]) -> list[str]:
# 	citations: list[str] = []
# 	for chunk in chunks:
# 		page = chunk.get("page")
# 		section = chunk.get("section")
# 		location_parts: list[str] = []
# 		if page is not None:
# 			location_parts.append(f"page {page}")
# 		if section:
# 			location_parts.append(f"section {section}")
# 		location = ", ".join(location_parts) if location_parts else "source location unavailable"
# 		citations.append(f"{location}")
# 	return citations


def parse_json_object(text: str) -> dict[str, Any]:
	try:
		return json.loads(text)
	except Exception:
		match = re.search(r"\{[\s\S]*\}", text)
		if match:
			try:
				return json.loads(match.group(0))
			except Exception:
				return {}
	return {}


def extract_from_images(image_urls: list[str], claim_category: str, claim_description: str) -> list[str]:
	"""Use Gemini multimodal endpoint to extract claim-relevant evidence text from images."""
	if not image_urls:
		return []

	prompt = (
		"Extract key insurance-claim facts from the provided image(s). "
		"Prioritize: dates, names, flight numbers, amounts, diagnosis/events, and official statements. "
		"Keep output concise and factual.\n"
		f"Claim category: {claim_category}\n"
		f"Claim description: {claim_description}"
	)

	content_blocks: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
	for url in image_urls:
		content_blocks.append({"type": "image_url", "image_url": {"url": url}})

	try:
		raw = action_llm.invoke([HumanMessage(content=content_blocks)])
		text = extract_text_content(raw.content).strip()
		return [text] if text else []
	except Exception as exc:
		logger.warning("OCR extraction failed: %s", exc)
		return []


def escalation_webhook_stub(state: AgentState) -> None:
	"""Phase-1 placeholder for helpdesk webhook integration."""
	logger.warning(
		"Emergency escalation webhook stub triggered (thread locked). Reason: %s",
		state.get("escalation_reason", "unspecified"),
	)


def emergency_escalation_node(state: AgentState):
	log_node_start("emergency_escalation", state)

	current_time = time.time()
	last_activity = state.get("last_activity_timestamp")
	TIMEOUT_SECONDS = 30 * 60

	if last_activity and (current_time - last_activity > TIMEOUT_SECONDS):
		result = {
            "session_closed": True,
            "final_response": MESSAGES["SESSION_TIMEOUT"],
            "next_agent": "end",
            "messages": [AIMessage(content=MESSAGES["SESSION_TIMEOUT"])],
        }
		log_node_end("emergency_escalation", result)
		return result

	if state.get("session_closed"):
		result = {
			"final_response": MESSAGES["SESSION_CLOSED"],
			"next_agent": "end",
			"messages": [AIMessage(content=MESSAGES["SESSION_CLOSED"])],
		}
		log_node_end("emergency_escalation", result)
		return result

	user_input = normalize_user_text(state["messages"][-1].content)
	log_event("emergency_escalation.scan", user_input=user_input)

	if detect_emergency(user_input) and state.get("service_category") == "claim":
		escalation_reason = "Emergency keyword override"
		updated_state = {
			"session_closed": True,
			"escalation_reason": escalation_reason,
			"final_response": MESSAGES["EMERGENCY_ESCALATION"],
			"next_agent": "end",
			"messages": [AIMessage(content=MESSAGES["EMERGENCY_ESCALATION"])],
			"last_activity_timestamp": current_time,
		}
		escalation_webhook_stub({**state, **updated_state})
		log_node_end("emergency_escalation", updated_state)
		return updated_state

	result = {
		"next_agent": "router",
		"final_response": "",
		"last_activity_timestamp": current_time,
		}
	
	# --- Reset claim_stage if incoming request is an inquiry ---
	if state.get("service_category") == "inquiry":
		result["claim_stage"] = None

	log_node_end("emergency_escalation", result)
	return result


def router_orchestrator_node(state: AgentState):
    log_node_start("router", state)
    user_input = normalize_user_text(state["messages"][-1].content)
    
    if check_guardrails(user_input):
        result = {
            "final_response": MESSAGES["OUT_OF_SCOPE_FALLBACK"],
            "next_agent": "end",
            "messages": [AIMessage(content=MESSAGES["OUT_OF_SCOPE_FALLBACK"])],
        }
        return result

    # Perform advanced decomposition
    analysis = decompose_complex_query(user_input)
    intent = analysis.get("intent", "inquiry")
    scenarios = analysis.get("claim_scenarios", [])

    log_event("router.analysis_complete", intent=intent, found_scenarios=len(scenarios))

    if intent == "mixed" or len(scenarios) > 1:
        # Route to policy inquiry first with the isolated question block
        result = {
            "user_input": analysis.get("policy_inquiry") or "What is the total coverage of the plan?",
            "pending_claim_transition": True,
            "pending_claim_scenarios": scenarios,
            "next_agent": "policy",
            "final_response": "",
        }
        log_node_end("router", result)
        return result

    if intent == "claim" or state.get("service_category") == "claim":
        result = {
            "next_agent": "claims_action",
            "policy_tier": state.get("policy_tier") or HARD_CODED_POLICY_TIER,
            "final_response": "",
        }
        return result

    result = {
        "next_agent": "policy",
        "policy_tier": state.get("policy_tier") or HARD_CODED_POLICY_TIER,
        "final_response": "",
    }
    return result


def policy_inquiry_node(state: AgentState):
	log_node_start("policy", state)
	query = state.get("user_input") or normalize_user_text(state["messages"][-1].content)

	if query.strip().lower() == "inquiry service":
		prompt_message = MESSAGES["INQUIRY_START"]
		result = {
			"final_response": prompt_message,
			"next_agent": "end",
			"messages": [AIMessage(content=prompt_message)],
		}
		log_event("policy.initial_inquiry_prompt")
		log_node_end("policy", result)
		return result

	policy_tier = state.get("policy_tier") or HARD_CODED_POLICY_TIER
	log_event("policy.retrieval_start", policy_tier=policy_tier, query=query)

	chunks = query_policy_chunks(query=query, top_k=5, policy_tier=policy_tier)
	max_score = chunks[0]["score"] if chunks else 0.0
	log_event("policy.retrieval_complete", matches=len(chunks), max_score=max_score)

	if not chunks or max_score < 0.35:
		result = {
			"final_response": MESSAGES["NO_REFERENCE_FALLBACK"],
			"next_agent": "end",
			"messages": [AIMessage(content=MESSAGES["NO_REFERENCE_FALLBACK"])],
		}
		log_event("policy.no_reference_fallback")
		log_node_end("policy", result)
		return result

	# citations = format_citations(chunks)
	context = "\n\n".join(chunk.get("text", "") for chunk in chunks if chunk.get("text"))
	# log_event("policy.context_prepared", citations=citations)
	log_event("policy.context_prepared")

	if state.get("service_category") == "claim" or state.get("claim_stage") == "ready_for_audit":
		result = {
			"policy_context": context,
			# "citations": citations,
			"next_agent": "claim_validation",
			"final_response": "",
		}
		log_event("policy.route_claim_validation")
		log_node_end("policy", result)
		return result

	system_prompt = (
		"You are a customer service assistant for travel insurance policy inquiries. Be concise and empathatic. "
		"Answer only from the provided context. Keep answer to 2-3 sentences. "
		# "Include citation references at the end using this format: [page x, section y]. " 
		"if context is not related to travel insurance policy, reply exactly with: "
		f"{MESSAGES["OUT_OF_SCOPE_FALLBACK"]}\n\n"
		"If context is insufficient, reply exactly with: "
		f"{MESSAGES["NO_REFERENCE_FALLBACK"]}\n\n"
		f"Context:\n{context}\n\n"
		f"User Question: {query}"
	)
	raw = policy_llm.invoke([HumanMessage(content=system_prompt)])
	answer = extract_text_content(raw.content).strip()
	log_event("policy.answer_generated", answer=answer)

	result = {
		"final_response": answer,
		"next_agent": "end",
		"messages": [AIMessage(content=answer)],
	}
	
	if state.get("pending_claim_transition"):
		scenarios = state.get("pending_claim_scenarios") or []
        
        # Create a scannable preview of the items the system broke down
		breakdown_text = "\n\nI can help you file a claim if you need it.\n"
		answer = f"{answer}{breakdown_text}"
        
		result = {
            "final_response": answer,
            "next_agent": "end",
            "claim_stage": "awaiting_claim_category",
            "service_category": "inquiry",
            # Pop or stage the first scenario for claims_action to process next turn
            "user_input": scenarios[0] if scenarios else "", 
            "pending_claim_transition": False,
            "messages": [AIMessage(content=answer)],
        }
	
	log_node_end("policy", result)
	return result


def claims_action_node(state: AgentState):
	log_node_start("claims_action", state)

	# 1. Pull existing values from state
	claim_category = state.get("claim_category")
	claim_description = (state.get("claim_description") or "").strip()
	image_urls = state.get("image_urls") or []

	# 2. Extract the latest raw text the user just typed
	latest_user_text = normalize_user_text(state["messages"][-1].content).strip()
	current_stage = state.get("claim_stage")

	# 3. Slot filling: Map the message text to the parameter we are currently awaiting
	if current_stage == "awaiting_claim_category" and not claim_category:
		claim_category = latest_user_text
		log_event("claims_action.slot_filled", slot="claim_category", value=claim_category)
		
	elif current_stage == "awaiting_description" and not claim_description:
		claim_description = latest_user_text
		log_event("claims_action.slot_filled", slot="claim_description", value=claim_description)

	log_event(
		"claims_action.slot_check",
		claim_category=claim_category,
		claim_description_present=bool(claim_description),
		image_count=len(image_urls),
	)

	# 4. Sequential validation gates (Always return filled slots so LangGraph saves them)
	if not claim_category:
		result = {
			"claim_stage": "awaiting_claim_category",
			"final_response": MESSAGES["CLAIM_CATEGORY"],
			"next_agent": "end",
			"messages": [AIMessage(content=MESSAGES["CLAIM_CATEGORY"])],
		}
		log_event("claims_action.awaiting_claim_category")
		log_node_end("claims_action", result)
		return result

	if not claim_description:
		result = {
			"claim_stage": "awaiting_description",
			"final_response": MESSAGES["CLAIM_DESCRIPTION"],
			"next_agent": "end",
			"messages": [AIMessage(content=MESSAGES["CLAIM_DESCRIPTION"])],
			"claim_category": claim_category,
		}
		log_event("claims_action.awaiting_description")
		log_node_end("claims_action", result)
		return result

	if not image_urls:
		result = {
			"claim_stage": "awaiting_images",
			"final_response": MESSAGES["CLAIM_IMAGE_PROOF"],
			"next_agent": "end",
			"messages": [AIMessage(content=MESSAGES["CLAIM_IMAGE_PROOF"])],
			"claim_category": claim_category,
			"claim_description": claim_description,
		}
		log_event("claims_action.awaiting_images")
		log_node_end("claims_action", result)
		return result

	# 5. Process images if all text slots are safely filled
	log_event("claims_action.ocr_start", image_count=len(image_urls))
	ocr_extracts = extract_from_images(image_urls, claim_category, claim_description)
	if not ocr_extracts:
		clarification = (
			"I could not extract enough details from your image attachments. "
			"Please upload clearer images or add key details in text."
		)
		result = {
			"claim_stage": "awaiting_images",
			"final_response": clarification,
			"next_agent": "end",
			"messages": [AIMessage(content=clarification)],
			"claim_category": claim_category,
			"claim_description": claim_description,
		}
		log_event("claims_action.ocr_insufficient")
		log_node_end("claims_action", result)
		return result

	retrieval_query = (
		f"Claim category: {claim_category}. Description: {claim_description}. "
		f"Evidence: {' '.join(ocr_extracts)}"
	)
	log_event("claims_action.ready_for_policy_audit", retrieval_query=retrieval_query)

	result = {
		"ocr_extracts": ocr_extracts,
		"user_input": retrieval_query,
		"claim_stage": "ready_for_audit",
		"next_agent": "policy",
		"final_response": "",
		"claim_category": claim_category,
		"claim_description": claim_description,
	}
	log_node_end("claims_action", result)
	return result


def claim_validation_node(state: AgentState):
	log_node_start("claim_validation", state)
	claim_category = state.get("claim_category", "")
	claim_description = state.get("claim_description", "")
	ocr_extracts = state.get("ocr_extracts") or []
	policy_context = state.get("policy_context", "")
	# citations = state.get("citations") or []
	attempts = int(state.get("clarification_attempts", 0))
	log_event(
		"claim_validation.audit_start",
		claim_category=claim_category,
		description_length=len(claim_description.strip()),
		ocr_extract_count=len(ocr_extracts),
		policy_context_present=bool(policy_context.strip()),
		clarification_attempts=attempts,
	)

	unclear_reasons: list[str] = []
	if len(claim_description.strip()) < 20:
		unclear_reasons.append("case description is too short")
	if not ocr_extracts:
		unclear_reasons.append("no extractable document evidence found")
	if not policy_context.strip():
		unclear_reasons.append("no matching policy clauses were retrieved")

	if unclear_reasons:
		attempts += 1
		log_event("claim_validation.unclear", reasons=unclear_reasons, attempts=attempts)
		if attempts > 2:
			result = {
				"clarification_attempts": 0,
				"claim_stage": "reset",
				"claim_category": "",
				"claim_description": "",
				"image_urls": [],
				"ocr_extracts": [],
				"policy_context": "",
				# "citations": [],
				"audit_report": {
					"status": "Flagged_For_Review",
					"reason": "Exceeded clarification retries",
				},
				"final_response": MESSAGES["CLAIM_RESET"],
				"next_agent": "end",
				"messages": [AIMessage(content=MESSAGES["CLAIM_RESET"])],
			}
			log_event("claim_validation.reset_after_retries")
			log_node_end("claim_validation", result)
			return result

		clarification_message = (
			"I need clarification before validating this claim: "
			+ "; ".join(unclear_reasons)
			+ ". Please provide the missing details and I will re-audit your case."
		)
		result = {
			"clarification_attempts": attempts,
			"claim_stage": "awaiting_clarification",
			"audit_report": {
				"status": "Flagged_For_Review",
				"reason": "; ".join(unclear_reasons),
				"clarification_attempts": attempts,
			},
			"final_response": clarification_message,
			"next_agent": "end",
			"messages": [AIMessage(content=clarification_message)],
		}
		log_event("claim_validation.request_clarification", message=clarification_message)
		log_node_end("claim_validation", result)
		return result

	validation_prompt = (
		"You are a claim validation auditor. Return JSON only with keys: "
		"status, possible_coverage_amount, rationale, flagged_items. "
		"status must be either Passed or Flagged_For_Review.\n"
		f"Plan name: {HARD_CODED_POLICY_TIER}\n"
		f"Prepayment option: {HARD_CODED_PREPAYMENT_OPTION}\n"
		f"Claim category: {claim_category}\n"
		f"Case description: {claim_description}\n"
		f"Extracted evidence: {' '.join(ocr_extracts)}\n"
		f"Policy snippets: {policy_context[:6000]}\n"
	)

	raw = validation_llm.invoke([HumanMessage(content=validation_prompt)])
	parsed = parse_json_object(extract_text_content(raw.content))
	log_event("claim_validation.model_response", parsed=parsed)

	status = parsed.get("status", "Flagged_For_Review")
	if status not in {"Passed", "Flagged_For_Review"}:
		status = "Flagged_For_Review"

	possible_coverage_amount = parsed.get("possible_coverage_amount", "Not determined")
	rationale = parsed.get("rationale", "Validation completed with limited details.")
	flagged_items = parsed.get("flagged_items", [])
	if not isinstance(flagged_items, list):
		flagged_items = [str(flagged_items)]

	report = {
		"status": status,
		"policy_tier": HARD_CODED_POLICY_TIER,
		"prepayment_option": HARD_CODED_PREPAYMENT_OPTION,
		"possible_coverage_amount": possible_coverage_amount,
		"rationale": rationale,
		"flagged_items": flagged_items,
		# "citations": citations,
	}

	response_text = (
		f"Validation result: {status}. Possible coverage amount: {possible_coverage_amount}. "
		f"Reason: {rationale}."
		# f"Policy references: {', '.join(citations) if citations else 'not available'}."
	)
	log_event("claim_validation.completed", status=status, possible_coverage_amount=possible_coverage_amount)

	result = {
		"clarification_attempts": 0,
		"claim_stage": "completed" if status == "Passed" else "awaiting_clarification",
		"audit_report": report,
		"final_response": response_text,
		"next_agent": "end",
		"messages": [AIMessage(content=response_text)],
	}
	log_node_end("claim_validation", result)
	return result


def route_after_emergency(state: AgentState):
	log_event(
		"route.after_emergency",
		next_agent=state.get("next_agent"),
		final_response_present=bool(state.get("final_response")),
		session_closed=bool(state.get("session_closed", False)),
	)
	if state.get("next_agent") == "end" or state.get("final_response"):
		return "end"
	return "router"


def route_from_router(state: AgentState):
	log_event("route.from_router", next_agent=state.get("next_agent"))
	return state.get("next_agent", "end")


def route_after_policy(state: AgentState):
	log_event("route.after_policy", next_agent=state.get("next_agent"))
	return state.get("next_agent", "end")


def route_after_claim_action(state: AgentState):
	log_event("route.after_claim_action", next_agent=state.get("next_agent"))
	return state.get("next_agent", "end")


workflow = StateGraph(AgentState)

workflow.add_node("emergency_escalation", emergency_escalation_node)
workflow.add_node("router", router_orchestrator_node)
workflow.add_node("policy", policy_inquiry_node)
workflow.add_node("claims_action", claims_action_node)
workflow.add_node("claim_validation", claim_validation_node)

workflow.add_edge(START, "emergency_escalation")

workflow.add_conditional_edges(
	"emergency_escalation",
	route_after_emergency,
	{
		"router": "router",
		"end": END,
	},
)

workflow.add_conditional_edges(
	"router",
	route_from_router,
	{
		"policy": "policy",
		"claims_action": "claims_action",
		"end": END,
	},
)

workflow.add_conditional_edges(
	"policy",
	route_after_policy,
	{
		"claim_validation": "claim_validation",
		"end": END,
	},
)

workflow.add_conditional_edges(
	"claims_action",
	route_after_claim_action,
	{
		"policy": "policy",
		"end": END,
	},
)

workflow.add_edge("claim_validation", END)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
	import uuid

	session_thread = str(uuid.uuid4())
	config = {"configurable": {"thread_id": session_thread}}

	print("\nInteractive LangGraph Multi-Agent Test Environment")
	print(f"Session Started (Thread ID: {session_thread})")
	print("Type prompt and press Enter. Type 'exit' or 'quit' to stop.")
	print("=" * 60)

	while True:
		try:
			user_prompt = input("\nYou: ")
			if user_prompt.strip().lower() in ["exit", "quit"]:
				print("\nSession ended. Goodbye!")
				break

			if not user_prompt.strip():
				continue

			output = app_graph.invoke(
				{
					"messages": [HumanMessage(content=user_prompt)],
					"user_input": user_prompt,
				},
				config=config,
			)
			print(f"Agent: {output['final_response']}")
		except KeyboardInterrupt:
			print("\nSession interrupted. Goodbye!")
			break
		except Exception as exc:
			print(f"Error executing workflow: {exc}")
