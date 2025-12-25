import os
import time
import cv2
import threading
import torch

from core.food_detector import detect_food_from_camera as detect_food
from core.video_generator import generate_animal_video
from config import (
    VIDEO_WINDOW_NAME,
    DETECTION_INTERVAL,
    DELETE_TEMP_VIDEO
)

class VideoPlayer:
    def __init__(self, window_name=VIDEO_WINDOW_NAME):
        self.window_name = window_name
        self.is_playing = False
        self.stop_requested = False

    def play(self, video_path):
        if not os.path.exists(video_path):
            print(f"[Error] Video file not found: {video_path}")
            return

        print(f"[Playing] {video_path}")
        self.is_playing = True
        self.stop_requested = False

        thread = threading.Thread(target=self._play_video, args=(video_path,))
        thread.daemon = True
        thread.start()

    def _play_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] Failed to open video: {video_path}")
            self.is_playing = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, width, height)

        while cap.isOpened() and not self.stop_requested:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow(self.window_name, frame)
            delay = max(1, int(1000 / fps))
            key = cv2.waitKey(delay)
            if key in (27, ord('q'), ord('Q')):
                self.stop_requested = True
                break

        cap.release()
        cv2.destroyWindow(self.window_name)
        self.is_playing = False
        print("[Stopped] Video playback finished.")

    def stop(self):
        if self.is_playing:
            self.stop_requested = True
            print("[Stopping] Stopping video...")

def main():
    print("=" * 50)
    print("Pet Dining AI Video System - Local v1.0")
    print("=" * 50)
    print("[Device] Running on:", "GPU" if torch.cuda.is_available() else "CPU")

    player = VideoPlayer()

    try:
        while True:
            print("\n[Status] Waiting for food detection...")

            food_name = detect_food()
            print(f"[Detected] Food: {food_name}")

            print("[Generating] Creating animal eating video...")
            start_time = time.time()
            video_path = generate_animal_video(food_name)
            elapsed = time.time() - start_time
            print(f"[Done] Video generated in {elapsed:.1f} seconds")

            print("[Playing] Launching video player...")
            player.play(video_path)

            while player.is_playing:
                time.sleep(0.5)

            if DELETE_TEMP_VIDEO and os.path.exists(video_path):
                os.remove(video_path)
                print(f"[Cleanup] Deleted temp video: {video_path}")

            print("[Waiting] Next detection in a few seconds...")
            time.sleep(DETECTION_INTERVAL)

    except KeyboardInterrupt:
        print("\n[Interrupted] User stopped the program.")
        player.stop()
    except Exception as e:
        print(f"[Error] {str(e)}")
        player.stop()
    finally:
        print("\n" + "=" * 50)
        print("Thank you for using the Pet Dining AI System!")
        print("=" * 50)

if __name__ == "__main__":
    try:
        import torch
    except ImportError:
        print("[Error] PyTorch is not installed. Run: pip install torch torchvision")
        exit(1)

    main()
