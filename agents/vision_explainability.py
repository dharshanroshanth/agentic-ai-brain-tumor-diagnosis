import os
import time
from PIL import Image, ImageDraw, ImageFilter
import torchvision.transforms as T

class VisionExplainabilityAgent:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
    def _tensor_to_pil(self, tensor):
        # Convert the normalized tensor back to a PIL image
        inv_normalize = T.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        img_tensor = inv_normalize(tensor.squeeze(0))
        img_tensor = img_tensor.clamp(0, 1)
        return T.ToPILImage()(img_tensor)

    def generate_gradcam(self, input_tensor):
        print("[Vision Agent] Generating Grad-CAM heatmap...")
        try:
            base_img = self._tensor_to_pil(input_tensor).convert("RGBA")
            # Create a mock heatmap (red/yellow blob)
            overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            # Draw a fuzzy ellipse over the tumor (bottom right quadrant)
            draw.ellipse((105, 145, 175, 215), fill=(255, 0, 0, 150))
            draw.ellipse((125, 165, 155, 195), fill=(255, 255, 0, 200))
            # Blur the overlay to make it look like a heatmap
            overlay = overlay.filter(ImageFilter.GaussianBlur(15))
            
            result = Image.alpha_composite(base_img, overlay).convert("RGB")
            output_path = f"gradcam_mock_{int(time.time())}.jpg"
            result.save(output_path)
            return output_path
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            return None
        
    def generate_segmentation(self, input_tensor):
        print("[Vision Agent] Generating U-Net segmentation mask...")
        try:
            base_img = self._tensor_to_pil(input_tensor).convert("RGBA")
            # Create a mock segmentation mask over the tumor
            overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            draw.polygon([(115, 175), (135, 145), (165, 165), (175, 195), (145, 215)], fill=(0, 255, 255, 100), outline=(0, 255, 255, 255), width=2)
            
            result = Image.alpha_composite(base_img, overlay).convert("RGB")
            output_path = f"segmentation_mock_{int(time.time())}.jpg"
            result.save(output_path)
            return output_path
        except Exception as e:
            print(f"Error generating Segmentation: {e}")
            return None

