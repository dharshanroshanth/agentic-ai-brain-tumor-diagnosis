import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from agents.perception import EnsemblePerceptionAgent, safety_gate
from agents.vision_explainability import VisionExplainabilityAgent
from agents.orchestrator import OrchestratorAgent
from agents.verifier import VerifierAgent
from agents.report_generator import ReportGenerationAgent

CLASSES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

class DiagnosticPipeline:
    def __init__(self):
        print("Initializing Agents...")
        self.perception = EnsemblePerceptionAgent(num_classes=4)
        
        # Load weights (it will gracefully skip models that aren't trained yet)
        self.perception.load_weights(
            effnet_path="checkpoints/best_fold1.pth",
            densenet_path="checkpoints/best_densenet_local.pth",
            vit_path="checkpoints/best_vit_local.pth"
        )
        self.perception.eval()
        
        # We need the last conv layer for Grad-CAM. Since EfficientNet is the CNN, we use it for explainability.
        target_layer = self.perception.effnet.backbone.features[-1]
        self.vision = VisionExplainabilityAgent(self.perception, target_layer)
        
        self.orchestrator = OrchestratorAgent()
        self.verifier = VerifierAgent()
        self.reporter = ReportGenerationAgent()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def run_case(self, case_id: str, image_path: str):
        print(f"\n--- Starting Case {case_id} ---")
        
        # 1. Perception
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load image: {e}")
            # Mock image for testing if path fails
            img = Image.new('RGB', (224, 224))
            
        input_tensor = self.transform(img).unsqueeze(0)
        
        with torch.no_grad():
            # Apply Test-Time Augmentation (TTA) for maximum accuracy
            outputs = []
            outputs.append(self.perception(input_tensor))
            outputs.append(self.perception(TF.hflip(input_tensor)))
            outputs.append(self.perception(TF.rotate(input_tensor, 15)))
            outputs.append(self.perception(TF.rotate(input_tensor, -15)))
            logits = torch.stack(outputs).mean(dim=0)
            
        probs = torch.softmax(logits, dim=-1)[0]
        
        # Apply Glioma Threshold trick
        glioma_idx = CLASSES.index("glioma")
        if probs[glioma_idx].item() > 0.30:
            pred_idx = torch.tensor(glioma_idx)
            max_prob = probs[glioma_idx]
        else:
            max_prob, pred_idx = torch.max(probs, dim=0)
            
        diagnosis = CLASSES[pred_idx.item()]
        confidence = max_prob.item()
        
        # 2. Uncertainty Gate
        decision = safety_gate(logits, entropy_threshold=0.8)[0]
        print(f"Perception output: {diagnosis} (Conf: {confidence:.2f}) -> Safety Gate: {decision}")
        
        # 3. Initialize Orchestrator State
        self.orchestrator.initialize_case(case_id, diagnosis, confidence, decision)
        
        # 4. Optional Vision tasks (Simulated as requested by orchestrator/rules)
        # If low confidence or requested, generate artifacts
        has_explainability = False
        gradcam_path = None
        segmentation_path = None
        
        if decision == "ESCALATE_TO_HUMAN" or confidence < 0.9:
            print("Invoking Vision Agents for explainability...")
            gradcam_path = self.vision.generate_gradcam(input_tensor)
            segmentation_path = self.vision.generate_segmentation(input_tensor)
            has_explainability = True
            self.orchestrator.update_state(has_segmentation=True, has_explainability=True)
            
        # 5. Verifier (Cross-check)
        print("Invoking Verifier Agent...")
        verification = self.verifier.verify_diagnosis(diagnosis)
        print(f"Verifier Status: {verification['status']}")
        
        # 6. Radiologist Chat Simulation
        print("--- Radiologist Chat Active ---")
        if decision == "ESCALATE_TO_HUMAN":
            radiologist_input = "The model seems uncertain. Does the verifier agree?"
        else:
            radiologist_input = "Looks good. Generate report."
            
        ai_response = self.orchestrator.chat(radiologist_input)
        print(f"Radiologist: {radiologist_input}")
        print(f"AI: {ai_response}")
        
        radiologist_notes = "Reviewed images and AI notes. Agree with findings."
        
        # 7. Final Report
        all_probs = {CLASSES[i]: probs[i].item() for i in range(len(CLASSES))}
        report = self.reporter.generate_report(
            case_id, diagnosis, confidence, decision, 
            verification, radiologist_notes, has_explainability, all_probs
        )
        
        print(f"\n{report}")
        return {
            "report": report,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "decision": decision,
            "all_probs": all_probs,
            "has_explainability": has_explainability,
            "gradcam_path": gradcam_path,
            "segmentation_path": segmentation_path
        }

if __name__ == "__main__":
    import sys
    # Use image path from arguments if provided, else use mock
    image_path = sys.argv[1] if len(sys.argv) > 1 else "mock_image.jpg"
    pipeline = DiagnosticPipeline()
    pipeline.run_case("CASE-101", image_path)
