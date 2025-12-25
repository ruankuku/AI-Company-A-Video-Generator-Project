import os
import time
import torch
from PIL import Image
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_video
from .image_selector import select_random_animal_image
from config import (
    MODEL_CACHE_DIR,
    ANIMAL_IMAGE_DIR,
    VIDEO_OUTPUT_DIR,
    FPS,
    VIDEO_DURATION,
    SEED,
    PROMPT_TEMPLATE,
    NEGATIVE_PROMPT,
    VIDEO_STEPS,
    VIDEO_GUIDANCE_SCALE,
    VIDEO_STRENGTH,
    BASE_MODEL,
    ADAPTER_MODEL
)


class LocalVideoGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = None
        self.model_loaded = False

    def load_model(self):
        if self.model_loaded:
            return

        print("Loading AnimateDiff model...")
        start_time = time.time()

        try:
            adapter = MotionAdapter.from_pretrained(
                ADAPTER_MODEL,
                cache_dir=MODEL_CACHE_DIR
            )

            self.pipe = AnimateDiffPipeline.from_pretrained(
                BASE_MODEL,
                motion_adapter=adapter,
                torch_dtype=self.dtype,
                cache_dir=MODEL_CACHE_DIR
            )

            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

            if self.device == "cuda":
                self.pipe.enable_vae_slicing()
                self.pipe.enable_model_cpu_offload()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    print("XFormers not available. Continuing without memory optimization.")

            self.model_loaded = True
            print(f"Model loaded in {time.time() - start_time:.1f}s")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def generate_video(self, animal_image_path: str, food_name: str, duration: int = None) -> str:
        if not self.model_loaded:
            self.load_model()

        init_image = Image.open(animal_image_path).convert("RGB")
        animal_type = self.extract_animal_type(animal_image_path)

        prompt = PROMPT_TEMPLATE.format(animal_type=animal_type, food_name=food_name)
        num_frames = FPS * (duration or VIDEO_DURATION)

        print(f"Generating video with prompt: {prompt}")

        generator = torch.Generator(device=self.device).manual_seed(SEED)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=init_image,
            strength=VIDEO_STRENGTH,
            num_inference_steps=VIDEO_STEPS,
            guidance_scale=VIDEO_GUIDANCE_SCALE,
            num_frames=num_frames,
            generator=generator,
        )

        output_path = self._build_output_path(animal_type, food_name)
        os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
        export_to_video(result.frames[0], output_path, fps=FPS)

        print(f"Video saved to: {output_path}")
        return output_path

    def extract_animal_type(self, image_path: str) -> str:
        parts = image_path.split(os.sep)
        return parts[-2] if len(parts) > 1 else "animal"

    def _build_output_path(self, animal: str, food: str) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{animal}_{food}_{timestamp}".lower().replace(" ", "_")
        return os.path.join(VIDEO_OUTPUT_DIR, f"{filename}.mp4")


def generate_animal_video(food_name: str) -> str:
    generator = LocalVideoGenerator()
    image_path = select_random_animal_image(ANIMAL_IMAGE_DIR)
    print(f"Selected animal image: {image_path}")
    return generator.generate_video(image_path, food_name)
