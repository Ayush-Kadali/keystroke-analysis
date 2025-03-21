import json
import pandas as pd
from datetime import datetime
import numpy as np

def process_keystroke_data(input_file, output_file, idle_threshold=2.0):
    """
    Process raw keystroke data and prepare it for analysis.
    
    Parameters:
    - input_file: Path to the raw keyboard events file
    - output_file: Path to save the processed data
    - idle_threshold: Time in seconds to consider as idle period (default: 2.0)
    """
    # Read raw data
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse JSON data
    events = []
    for line in lines:
        try:
            event = json.loads(line.strip())
            events.append(event)
        except json.JSONDecodeError:
            continue
    
    # Create a dictionary to track key states
    key_states = {}
    processed_events = []
    
    for event in events:
        key_name = event['name']
        event_type = event['event_type']
        timestamp = event['time']
        
        # Handle multiple down events for the same key
        if event_type == 'down':
            if key_name in key_states and key_states[key_name]['state'] == 'down':
                # Key was already down, update the timestamp
                key_states[key_name]['down_time'] = timestamp
            else:
                # New key press
                key_states[key_name] = {
                    'state': 'down',
                    'down_time': timestamp,
                    'scan_code': event['scan_code'],
                    'is_keypad': event['is_keypad'],
                    'device': event['device']
                }
        
        # Handle key release
        elif event_type == 'up':
            if key_name in key_states and key_states[key_name]['state'] == 'down':
                # Calculate dwell time (how long the key was pressed)
                dwell_time = timestamp - key_states[key_name]['down_time']
                
                # Store the complete key press event
                processed_events.append({
                    'key': key_name,
                    'scan_code': key_states[key_name]['scan_code'],
                    'down_time': key_states[key_name]['down_time'],
                    'up_time': timestamp,
                    'dwell_time': dwell_time,  # Hold Time in your diagram
                    'is_keypad': key_states[key_name]['is_keypad'],
                    'device': key_states[key_name]['device']
                })
                
                # Update key state
                key_states[key_name]['state'] = 'up'
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(processed_events)
    
    if len(df) == 0:
        print("No valid keystroke events found!")
        return
    
    # Sort by down_time
    df = df.sort_values('down_time')
    
    # Calculate all four keystroke dynamics metrics
    # 1. Hold Time / Dwell Time - already calculated above
    
    # 2. Press-Press Time (time between consecutive key presses)
    df['next_down_time'] = df['down_time'].shift(-1)
    df['press_press_time'] = df['next_down_time'] - df['down_time']
    
    # 3. Release-Release Time (time between consecutive key releases)
    df['next_up_time'] = df['up_time'].shift(-1)
    df['release_release_time'] = df['next_up_time'] - df['up_time']
    
    # 4. Release-Press Time (time between releasing one key and pressing the next)
    df['release_press_time'] = df['next_down_time'] - df['up_time']
    
    # Filter out idle periods
    df['is_idle'] = df['press_press_time'] > idle_threshold
    
    # Mark typing sessions (sequences without idle periods)
    session_ids = (df['is_idle'].shift(1).fillna(False)).cumsum()
    df['session_id'] = session_ids
    
    # Remove very short sessions (less than 5 keystrokes)
    session_counts = df['session_id'].value_counts()
    valid_sessions = session_counts[session_counts >= 5].index
    df = df[df['session_id'].isin(valid_sessions)]
    
    # Generate timestamp for human-readable reference
    df['timestamp'] = df['down_time'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S.%f'))
    
    # Calculate features for each session
    session_features = []
    
    for session_id, session_data in df.groupby('session_id'):
        # Skip very short sessions
        if len(session_data) < 5:
            continue
            
        # Calculate statistics for all four timing metrics
        timing_metrics = {
            'dwell_time': 'Hold Time',
            'press_press_time': 'Press-Press Time',
            'release_release_time': 'Release-Release Time',
            'release_press_time': 'Release-Press Time'
        }
        
        feature_stats = {}
        for metric, description in timing_metrics.items():
            # Filter out idle periods for inter-key metrics
            data = session_data[metric]
            if metric != 'dwell_time':  # Not for dwell time
                data = data[~session_data['is_idle']]
            
            if len(data) > 0:
                feature_stats[f'mean_{metric}'] = data.mean()
                feature_stats[f'std_{metric}'] = data.std()
                feature_stats[f'min_{metric}'] = data.min()
                feature_stats[f'max_{metric}'] = data.max()
        
        # Special key usage
        special_keys = ['shift', 'ctrl', 'alt', 'space', 'backspace', 'enter', 'tab']
        special_key_counts = {key: sum(session_data['key'] == key) for key in special_keys}
        special_key_ratio = sum(special_key_counts.values()) / len(session_data)
        
        # Store session features
        session_features.append({
            'session_id': session_id,
            'start_time': session_data['down_time'].min(),
            'end_time': session_data['up_time'].max(),
            'duration': session_data['up_time'].max() - session_data['down_time'].min(),
            'num_keystrokes': len(session_data),
            'special_key_ratio': special_key_ratio,
            **feature_stats,
            **special_key_counts
        })
    
    # Create session features DataFrame
    session_df = pd.DataFrame(session_features)
    
    # Add timestamp for readability
    session_df['start_timestamp'] = session_df['start_time'].apply(
        lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # Save processed data
    result = {
        'keystroke_events': df.to_dict('records'),
        'session_features': session_df.to_dict('records')
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Processed {len(df)} keystroke events across {len(session_df)} typing sessions.")
    print(f"Data saved to {output_file}")
    
    return df, session_df

# Example usage
if __name__ == "__main__":
    input_file = "events.txt"
    output_file = "processed_keystrokes.json"
    process_keystroke_data(input_file, output_file, idle_threshold=2.0)
