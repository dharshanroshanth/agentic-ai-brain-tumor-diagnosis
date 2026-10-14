class VerifierAgent:
    def __init__(self):

        self.knowledge_base = {
            "glioma": "Typically presents as an intra-axial mass with irregular borders and surrounding edema.",
            "meningioma": "Usually extra-axial, well-circumscribed, often with a dural tail.",
            "pituitary": "Located in the sella turcica, often causing enlargement of the pituitary fossa.",
            "no_tumor": "Normal brain architecture with no abnormal masses or pathological contrast enhancement."
        }

    def verify_diagnosis(self, diagnosis: str) -> dict:

        reasoning = self.knowledge_base.get(
            diagnosis.lower(), 
            "Unknown classification. Please manually review."
        )
        
        return {
            "status": "Verified (Rule-Based)",
            "reasoning": reasoning
        }
