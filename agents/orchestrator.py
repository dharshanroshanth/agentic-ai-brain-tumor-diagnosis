class OrchestratorAgent:
    def __init__(self):
        self.state = {}

    def initialize_case(self, case_id, diagnosis, confidence, decision):
        self.state = {
            "case_id": case_id,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "decision": decision,
            "has_segmentation": False,
            "has_explainability": False
        }
        print(f"[Orchestrator] Case {case_id} initialized.")

    def update_state(self, **kwargs):
        self.state.update(kwargs)
        print(f"[Orchestrator] State updated: {kwargs}")

    def chat(self, user_input: str) -> str:
        # Deterministic rule-based responses instead of an LLM
        if self.state.get("decision") == "ESCALATE_TO_HUMAN":
            return f"System notes uncertainty in the '{self.state.get('diagnosis')}' prediction. Standing by for human override."
        else:
            return "All metrics are within acceptable thresholds. Proceeding with report generation."
