from pynput import keyboard
import time
import pandas as pd
import os

# Store timing data
key_press_times = {}
key_release_times = {}
keystroke_data = []
user_id = input("Enter user ID: ")

def on_press(key):
    try:
        key_press_times[key] = time.time()
    except:
        pass

def on_release(key):
    try:
        if key in key_press_times:
            release_time = time.time()
            press_time = key_press_times[key]
            hold_duration = release_time - press_time
            
            # Record the previous key if available to calculate latency
            prev_key = None
            prev_release_time = None
            if keystroke_data:
                prev_key = keystroke_data[-1]['key']
                prev_release_time = keystroke_data[-1]['release_time']
            
            latency = None
            if prev_release_time:
                latency = press_time - prev_release_time
                
            keystroke_data.append({
                'user_id': user_id,
                'key': str(key),
                'press_time': press_time,
                'release_time': release_time,
                'hold_duration': hold_duration,
                'latency': latency,
                'previous_key': prev_key
            })
            
            # For demonstration purposes, stop after 200 keystrokes
            if len(keystroke_data) >= 200:
                save_data()
                return False
    except:
        pass
    
    # Stop on Esc key
    if key == keyboard.Key.esc:
        save_data()
        return False

def save_data():
    df = pd.DataFrame(keystroke_data)
    os.makedirs('keystroke_data', exist_ok=True)
    file_path = f'keystroke_data/{user_id}_{time.strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

# Start the listener
print("Type normally. Press Esc to stop recording.")
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
