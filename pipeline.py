from ultralytics import YOLO

def crop(res, img):
    """
    Crop the image based on the bounding box coordinates.
    The image is padded by 1% of the bounding box size.

    Args:
        res (list): List of bounding box coordinates.
        img (cv2): The original image.

    Returns:
        list: List of cropped images.
    """

    boxes = res[0].boxes.xyxy.cpu().numpy()  # bounding boxes

    cropped_images = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # Pad the crop by 1%
        height, width, _ = img.shape
        pad_x = int(0.01 * (x2 - x1))
        pad_y = int(0.01 * (y2 - y1))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)

        crop = img[y1:y2, x1:x2]
        cropped_images.append(crop)

    return cropped_images


def run_object_detection(image, source, conf=0.25, iou=0.45, device='0', show=True):
    """
    Run object detection using YOLOv8 model.

    Args:
        image (cv2): the current image in the input_dir.
        model_path (str): Path to the YOLOv8 model file.
        source (str): Source for inference (e.g., image, video, webcam).
        conf (float): Confidence threshold for predictions.
        iou (float): IoU threshold for NMS.
        device (str): Device to run the model on (e.g., '0' for GPU).
        show (bool): Whether to display the results.

    Returns:
        None
    """
    # Load the YOLOv8 model
    model = YOLO('saved_models\obj.pt')

    # Run inference
    results = model(image)

    cropped_objects = crop(results, image)
