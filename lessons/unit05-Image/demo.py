from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from torchvision.ops import nms


def main() -> None:
    image_path = Path(__file__).resolve().with_name("sample2.jpg")
    # other options: facebook/detr-resnet-50-dc5, or search in https://huggingface.co/models?search=detr
    model_name = "facebook/detr-resnet-50"
    target_classes = ["car", "cat"]
    image = Image.open(image_path).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(model_name)
    # Invoke __call__ mathod: return as PyTorch tensors
    inputs = processor(images=image, return_tensors="pt")

    with torch.inference_mode():
        outputs = model(**inputs)

    # After getting raw logits, this method will apply threshold and (optional) target size.
    detections = processor.post_process_object_detection(
        outputs, # contains raw logits for classification and predicted regions
        threshold=0.5,
        target_sizes=torch.tensor([image.size[::-1]]),
    )[0]

    # This step tries to identify all class ids which are recognized as labels in target_classes
    target_label_ids = {
        label_id: label
        for label_id, label in model.config.id2label.items()
        if label.lower() in target_classes
    }
    # Filter based on label ids
    objects = [
        (score.item(), box, target_label_ids[label_id.item()])
        for score, box, label_id in zip(
            detections["scores"],
            detections["boxes"],
            detections["labels"],
        )
        if label_id.item() in target_label_ids
    ]
    if objects:
        # Perform non-maximum suppression to remove (potentially) duplicated objects.
        # We might want to do this per class(?)
        keep_indices = nms(
            boxes=torch.stack([box for _, box, _ in objects]),
            scores=torch.tensor([score for score, _, _ in objects]),
            iou_threshold=0.5,
        ).tolist()
        objects = [(objects[index][0], objects[index][1].tolist(), objects[index][2]) for index in keep_indices]

    if not objects:
        print("No target objects detected.")
        return

    print(f"Detected {len(objects)} object(s):")
    for index, (score, (x_min, y_min, x_max, y_max), label_name) in enumerate(objects, start=1):
        print(
            f"{index}. confidence={score:.2f}, "
            f"box=(left={x_min:.0f}, top={y_min:.0f}, "
            f"right={x_max:.0f}, bottom={y_max:.0f}), "
            f"label_name: {label_name}"
        )
    # TODO: try to overlay the bboxes to the original image?


if __name__ == "__main__":
    main()