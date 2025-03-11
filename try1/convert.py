import json
import pandas as pd
import numpy as np
from collections import defaultdict

def process_keyboard_events(input_file, output_file, user_id):
    # Read the raw data
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse JSON events
    events = []
    for line in lines:
        try:
            # Some lines might have characters printed to console mixed with JSON
            # Find where the JSON starts (first '{')
            json_start = line.find('{')
            if json_start >= 0:
                line = line[json_start:]
                event = json.loads(line)
                events.append(event)
        except json.JSONDecodeError:
            # Skip lines that aren't valid JSON
            continue
    
    # Process events to extract keystroke dynamics
    key_states = defaultdict(lambda: {'first_down': None, 'last_down': None, 'up': None})
    keystroke_data = []
    
    # First pass: group events by key
    for event in events:
        key_name = event['name']
        event_type = event['event_type']
        timestamp = event['time']
        
        if event_type == 'down':
            # If this is the first down event for this key
            if key_states[key_name]['first_down'] is None:
                key_states[key_name]['first_down'] = timestamp
            # Always update last_down time
            key_states[key_name]['last_down'] = timestamp
        elif event_type == 'up':
            # Record the up event
            key_states[key_name]['up'] = timestamp
            
            # If we have both down and up events, record the keystroke
            if key_states[key_name]['first_down'] is not None:
                keystroke_data.append({
                    'user_id': user_id,
                    'key': key_name,
                    'press_time': key_states[key_name]['first_down'],
                    'release_time': timestamp,
                    'hold_duration': timestamp - key_states[key_name]['first_down'],
                    # Other fields will be filled in second pass
                    'latency': None,
                    'previous_key': None
                })
                
                # Reset this key's state
                key_states[key_name] = {'first_down': None, 'last_down': None, 'up': None}
    
    # Second pass: calculate latency between keystrokes
    keystroke_data.sort(key=lambda x: x['press_time'])
    
    for i in range(1, len(keystroke_data)):
        current = keystroke_data[i]
        previous = keystroke_data[i-1]
        
        # Calculate latency (time between previous key release and current key press)
        current['latency'] = current['press_time'] - previous['release_time']
        current['previous_key'] = previous['key']
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(keystroke_data)
    df.to_csv(output_file, index=False)
    
    print(f"Processed {len(keystroke_data)} keystrokes")
    print(f"Data saved to {output_file}")
    
    return df

# Example usage
user_id = "user1"  # You can change this or make it a parameter
df = process_keyboard_events('events.txt', f'keystroke_data_{user_id}.csv', user_id)

# Display some basic statistics
print("\nBasic Statistics:")
print(f"Average hold duration: {df['hold_duration'].mean():.4f} seconds")
if not df['latency'].isna().all():
    print(f"Average latency between keys: {df['latency'].dropna().mean():.4f} seconds")
most_common_keys = df['key'].value_counts().head(5)
print("\nMost common keys:")
for key, count in most_common_keys.items():
    print(f"  {key}: {count}")
