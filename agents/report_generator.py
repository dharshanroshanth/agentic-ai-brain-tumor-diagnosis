PROTOCOLS = {
    "glioma": {
        "urgency": "CRITICAL - Time-sensitive intervention required.",
        "guidelines": "Gliomas, particularly high-grade variants like glioblastoma multiforme (GBM), exhibit aggressive infiltration and rapid expansion. Standard of care includes maximal safe surgical resection followed by concurrent chemoradiation (Stupp protocol).",
        "measures": [
            "- STAT neurosurgical consultation for biopsy and resection planning.",
            "- Order contrast-enhanced MRI (T1 post-gadolinium) to assess blood-brain barrier breakdown.",
            "- Administer corticosteroids (e.g., Dexamethasone) immediately to reduce peritumoral edema and mass effect.",
            "- Schedule advanced functional imaging (fMRI/DTI) if the lesion is near eloquent cortex.",
            "- Prepare for molecular profiling (IDH mutation, MGMT promoter methylation) post-biopsy."
        ]
    },
    "meningioma": {
        "urgency": "MODERATE TO HIGH - Dependent on mass effect and size.",
        "guidelines": "Typically benign, slow-growing extra-axial tumors originating from the arachnoid cap cells. Management ranges from active surveillance for asymptomatic lesions to surgical resection for symptomatic/growing lesions.",
        "measures": [
            "- Schedule routine neurosurgical evaluation.",
            "- Assess clinically for symptoms of mass effect (headaches, visual field deficits, focal neurology).",
            "- If asymptomatic and small (< 2cm): Initiate 'wait-and-scan' protocol with follow-up MRI in 6 months.",
            "- If symptomatic: Prepare for surgical resection (Simpson grading applies).",
            "- Stereotactic radiosurgery (SRS) may be considered for surgically inaccessible lesions or high-risk patients."
        ]
    },
    "pituitary": {
        "urgency": "MODERATE - Assess for endocrine dysfunction and optic chiasm compression.",
        "guidelines": "Pituitary adenomas are typically benign neuroendocrine tumors. They are classified into functioning (hormone-secreting) and non-functioning. Primary concerns are visual loss and endocrinopathy.",
        "measures": [
            "- STAT visual field testing (perimetry) to check for bitemporal hemianopsia (optic chiasm compression).",
            "- Draw comprehensive endocrine panel (Prolactin, IGF-1, ACTH, Cortisol, TSH, Free T4, LH/FSH).",
            "- Initiate multi-disciplinary Endocrine and Neurosurgery consults.",
            "- If prolactinoma is suspected, initiate dopamine agonist therapy (e.g., Cabergoline) as first-line treatment.",
            "- If non-functioning with mass effect, prepare for transsphenoidal surgical resection."
        ]
    },
    "no_tumor": {
        "urgency": "LOW - Routine clinical follow-up.",
        "guidelines": "No anomalous intra-axial or extra-axial masses detected within the standard detection thresholds of the neural engine.",
        "measures": [
            "- Correlate imaging results with patient's presenting clinical symptoms.",
            "- If neurological symptoms persist, consider alternative etiologies (vascular, demyelinating, infectious, or metabolic).",
            "- Review scan for non-tumoral pathologies (e.g., microvascular ischemic changes).",
            "- Routine clinical follow-up as indicated by primary care provider."
        ]
    }
}

class ReportGenerationAgent:
    def __init__(self):
        pass

    def generate_report(self, case_id, diagnosis, confidence, decision, verification, radiologist_notes, has_explainability, all_probs=None) -> str:
        
        info = PROTOCOLS.get(diagnosis, PROTOCOLS["no_tumor"])
        measures_str = "\n".join(info['measures'])

        template = f"""
=== FINAL CLINICAL REPORT ===
{case_id}

[FINDINGS]
Primary AI Classification: {diagnosis.upper()}
Model Confidence Score: {confidence:.2%}

[CLINICAL URGENCY]
{info['urgency']}

[VERIFICATION & PATHOLOGY]
Expected visual indicators for {diagnosis}:
- {verification['reasoning']}

[CLINICAL GUIDELINES]
{info['guidelines']}

[RECOMMENDED MEASURES & NEXT STEPS]
{measures_str}

[SYSTEM STATUS]
Safety Gate Decision: {decision}
Explainability Modules Triggered: {'Yes (See Grad-CAM/Segmentation)' if has_explainability else 'No'}

[RADIOLOGIST NOTES]
{radiologist_notes}

[ASSESSMENT & PLAN]
Diagnosis of {diagnosis} logged into system.
Action required: {'Immediate human review and escalation' if decision == 'ESCALATE_TO_HUMAN' else 'Proceed with clinical guidelines above'}
===========================
"""
        return template.strip()
