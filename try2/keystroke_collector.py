from pynput import keyboard
import json
import time
import os
from datetime import datetime

class KeystrokeCollector:
    def __init__(self, output_dir="keystroke_data"):
        self.output_dir = output_dir
        self.current_session = []
        self.last_keystroke_time = None
        self.session_timeout = 2.0
        self.user_id = None
        self.session_id = None

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(f"{output_dir}/raw"):
            os.makedirs(f"{output_dir}/raw")
        if not os.path.exists(f"{output_dir}/processed"):
            os.makedirs(f"{output_dir}/processed")

    def start_collection(self, user_id="unknown"):
        self.user_id = user_id
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Starting keystroke collection for user: {user_id}")
        print("Press ESC to stop collection")

        with keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release) as listener:
            listener.join()

        print(f"Keystroke collection stopped for user: {user_id}")

    def _on_press(self, key):
        current_time = time.time()

        if self.last_keystroke_time is not None and \
           (current_time - self.last_keystroke_time) > self.session_timeout:
            self._save_current_session()
            self.current_session = []

        self.last_keystroke_time = current_time

        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key)

        self.current_session.append({
            "event_type": "down",
            "name": key_name,   # <-- Fix here
            "time": current_time
            })

        

    def _on_release(self, key):
        current_time = time.time()

        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key)

        self.current_session.append({
            "event_type": "up",
            "name": key_name,   # <-- Fix here
            "time": current_time
        })


        if key == keyboard.Key.esc:
            self._save_current_session()
            return False

    def _save_current_session(self):
        if len(self.current_session) < 10:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/raw/{self.user_id}_{self.session_id}_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.current_session, f, indent=2)

        print(f"Saved session with {len(self.current_session)} keystrokes to {filename}")

if __name__ == "__main__":
    user_id = input("Enter user ID: ")
    collector = KeystrokeCollector()
    collector.start_collection(user_id)
