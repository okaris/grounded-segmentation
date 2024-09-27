import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor
)
from transformers.image_utils import load_image
import argparse

class ObjectSegmenter:
    def __init__(self, detector_id="IDEA-Research/grounding-dino-tiny", segmenter_id="facebook/sam-vit-huge", device="cpu"):
        print(f"Initializing grounded segmenter with detector_id: {detector_id} and segmenter_id: {segmenter_id}")
        self.device = device
        self.detector_processor = AutoProcessor.from_pretrained(detector_id)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(self.device)

        self.segmenter_processor = SamProcessor.from_pretrained(segmenter_id)
        self.segmenter = SamModel.from_pretrained(segmenter_id).to(self.device)

    def to(self, device):
        self.device = device
        self.detector.to(device)
        self.segmenter.to(device)

    def segment(self, image: Image.Image, keyword: str, score_threshold: float = 0.7, box_threshold: float = 0.7, text_threshold: float = 0.7, resolution: int = 1024) -> np.ndarray:
        
        original_width, original_height = image.size
        width, height = image.size
        if width > height:
            new_width = resolution
            new_height = int(height * (resolution / width))
        else:
            new_height = resolution
            new_width = int(width * (resolution / height))
        image = image.resize((new_width, new_height))
        
        # Detect objects
        print("Detecting objects...")
        inputs = self.detector_processor(images=image, text=f"{keyword.lower()}.", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.detector(**inputs)

        results = self.detector_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_threshold, text_threshold=text_threshold, target_sizes=[image.size[::-1]]
        )

        if not results or len(results[0]["boxes"]) == 0:
            print("Segmentation failed. No results found.")
            return Image.new("L", image.size, 0)
        
        first_box = results[0]["boxes"][0].tolist()
        input_box = [[first_box[0], first_box[1]], [first_box[2], first_box[3]]]  # SAM expects top-right and bottom-left points

        # Segment the object
        print("Segmenting...")
        with torch.no_grad():
            inputs = self.segmenter_processor(image, input_boxes=[input_box], return_tensors="pt").to(self.device)
            outputs = self.segmenter(**inputs)

        masks = self.segmenter_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0][0]
        scores = outputs.iou_scores[0][0]

        best_mask_idx = scores.argmax().item()
        best_score = scores[best_mask_idx].item()
        
        if best_score < score_threshold:
            print("Segmentation failed. No results found.")
            return Image.new("L", image.size, 0)

        mask = masks[best_mask_idx].squeeze().numpy()
        mask = (mask * 255).astype(np.uint8)
        print("Finished segmenting.")
        
        mask = Image.fromarray(mask, mode='L')
        mask = mask.resize((original_width, original_height))
        return mask
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment objects in an image.")
    parser.add_argument("--image_url", type=str, required=True, help="URL of the image to segment")
    parser.add_argument("--keyword", type=str, required=True, help="Keyword for object detection")
    parser.add_argument("--dino_model", type=str, default="IDEA-Research/grounding-dino-tiny", help="Grounding DINO model to use")
    parser.add_argument("--sam_model", type=str, default="facebook/sam-vit-huge", help="SAM model to use")
    parser.add_argument("--output", type=str, default="mask.png", help="Output file name")
    args = parser.parse_args()

    segmenter = ObjectSegmenter(detector_id=args.dino_model, segmenter_id=args.sam_model)
    if torch.cuda.is_available():
        segmenter.to("cuda")
    else:
        print("CUDA is not available. Using CPU instead.")
        segmenter.to("cpu")
    
    image = load_image(args.image_url)
    
    mask = segmenter.segment(image=image, keyword=args.keyword)

    mask.save(args.output)
    print(f"Mask saved as {args.output}")

    segmenter.to("cpu")