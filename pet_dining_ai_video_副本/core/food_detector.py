import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import json
import os
import urllib.request
from config import FOOD_KEYWORDS, IMAGENET_LABELS_PATH

def load_imagenet_labels():
    if not os.path.exists(IMAGENET_LABELS_PATH):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        urllib.request.urlretrieve(url, IMAGENET_LABELS_PATH.replace(".json", ".txt"))
        with open(IMAGENET_LABELS_PATH.replace(".json", ".txt")) as f:
            classes = [line.strip() for line in f]
        with open(IMAGENET_LABELS_PATH, "w") as f:
            json.dump(classes, f)
    with open(IMAGENET_LABELS_PATH) as f:
        return json.load(f)

def detect_food_from_camera() -> str:
    print("🔍 Starting real-time food detection (press 'q' to exit)...")

    model = models.resnet50(pretrained=True)
    model.eval()
    labels = load_imagenet_labels()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    frame_count = 0
    last_detected_food = "food"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        if frame_count % 30 == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            top5 = torch.topk(probs, 5).indices.tolist()

            for idx in top5:
                label = labels[idx]
                if any(word in label.lower() for word in FOOD_KEYWORDS):
                    last_detected_food = label
                    break

        cv2.putText(display_frame, f"Detected: {last_detected_food}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Food Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final food detected: {last_detected_food}")
    return last_detected_food
