import re

COMPETITORS = ["manulife", "caa", "allianz", "sun life", "desjardins", "td insurance", "tugo"]
EMERGENCY_KEYWORDS = [
    "death",
    "dismemberment",
    "eye",
    "limb",
    "urgent help",
]

def check_guardrails(user_input: str) -> bool:
    """
    Returns True if a prompt injection signature or a competitor name is detected.
    """
    text_lower = user_input.lower()
    
    # 1. Competitor Check
    for competitor in COMPETITORS:
        if competitor in text_lower:
            return True
            
    # 2. Simple Prompt Injection Heuristics
    injection_patterns = [
        r"ignore previous instructions",
        r"system prompt",
        r"you are now a",
        r"bypass safety",
        r"output the hidden"
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            return True
            
    return False


def detect_emergency(user_input: str) -> bool:
    """Returns True when emergency intent should trigger live-agent escalation."""
    text_lower = user_input.lower()
    return any(keyword in text_lower for keyword in EMERGENCY_KEYWORDS)