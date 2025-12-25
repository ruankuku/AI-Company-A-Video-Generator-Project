
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
