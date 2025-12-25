# Data-Science

# AI Company: A Video Generator Project

This project simulates a pet (e.g., dog, cat, pig) mimicking a human by eating food. Using a webcam, the system identifies food items in real-time and generates a short video where a randomly chosen animal image "eats" the detected food using AnimateDiff and IP-Adapter.

## Core Features

1.Food recognition from webcam (ImageNet-based classification)

2.Random selection of animal images by category (dog, cat, etc.)

3.Video generation using AnimateDiff

4.Personalized prompt template for naturalistic, animated effects

5.Clean modular structure, easy to expand

## Project Structure
```
pet_dining_ai_video/
├── main.py
├── requirements.txt
├── animal_images/
│   ├── cat/
│   │   ├── cat1.jpg
│   │   └── cat2.jpg
│   ├── dog/
│   │   ├── dog1.jpg
│   │   ├── dog2.jpg
│   │   └── dog3.jpg
│   └── hamster/
│       ├── hamster1.jpg
│       └── hamster2.jpg
├── core/
│   ├── food_detector.py
│   ├── image_selector.py
│   └── video_generator.py
├── config.py
└── output
```

## Installation Steps

**1.Clone the repository:**
```
git clone git@github.com/ruankuku/AI-Company-A-Video-Generator-Project.git
```

**2.Create and activate a virtual environment:**
```
conda create --name base python=3.12.7
```

```
conda activate base
```

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python transformers diffusers xformers accelerate
```

**3.Install dependencies:**
```
pip install -r requirements.txt
```

**4.Prepare animal images:**
```
mkdir animal_images
mkdir animal_images/cat
mkdir animal_images/dog
mkdir animal_images/hamster
```

## Usage Instructions**
```
python main.py
```

## Configuration File Explanation**

Edit the config.py file to customize system behavior:
```

MODEL_CACHE_DIR = "models"
ADAPTER_MODEL = "guoyww/animatediff-motion-adapter-v1-5-2"
BASE_MODEL = "emilianJR/epiCRealism"

ANIMAL_IMAGE_DIR = "animal_images"
VIDEO_OUTPUT_DIR = "output_videos"
IMAGENET_LABELS_PATH = "imagenet_classes.json"

FPS = 24
VIDEO_DURATION = 5  
VIDEO_STRENGTH = 0.7
VIDEO_STEPS = 30
VIDEO_GUIDANCE_SCALE = 7.5
SEED = 42

PROMPT_TEMPLATE = (
    "A cute {animal_type} happily eating {food_name}, "
    "in a cozy home environment, soft lighting, detailed fur, cinematic animation, 4k resolution"
)
NEGATIVE_PROMPT = "bad quality, worse quality, deformed, distorted, text, watermark"

FOOD_KEYWORDS = [
    "sandwich", "burger", "cake", "ice cream", "pizza",
    "noodle", "pasta", "salad", "food", "sushi",
    "hotdog", "coffee", "tea", "fries", "apple", "banana"
]

VIDEO_WINDOW_NAME = "Pet Dining Experience"

DETECTION_INTERVAL = 3  
DELETE_TEMP_VIDEO = True
DEFAULT_ANIMAL_TYPE = "animal"
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png"]
```

**Let your pets dine with you in the virtual world !**
