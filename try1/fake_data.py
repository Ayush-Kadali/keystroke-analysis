import json
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def generate_user_keystroke_patterns(user_id, num_sessions=5, keystrokes_per_session=100):
    """
    Generate synthetic keystroke data for a specific user with consistent patterns.
    Each user will have slightly different timing characteristics.
    """
    # User-specific timing characteristics (milliseconds converted to seconds)
    user_profiles = {
        'user1': {
            'mean_dwell': 0.08,  # 80ms average hold time
            'dwell_std': 0.02,
            'mean_latency': 0.2,  # 200ms between keystrokes
            'latency_std': 0.05,
            'special_key_freq': 0.15  # 15% special keys
        },
        'user2': {
            'mean_dwell': 0.12,  # 120ms average hold time
            'dwell_std': 0.03,
            'mean_latency': 0.15,  # 150ms between keystrokes
            'latency_std': 0.04,
            'special_key_freq': 0.12
        },
        'user3': {
            'mean_dwell': 0.09,
            'dwell_std': 0.015,
            'mean_latency': 0.25,
            'latency_std': 0.06,
            'special_key_freq': 0.18
        },
        'user4': {
            'mean_dwell': 0.15,
            'dwell_std': 0.04,
            'mean_latency': 0.18,
            'latency_std': 0.03,
            'special_key_freq': 0.10
        },
        'user5': {
            'mean_dwell': 0.07,
            'dwell_std': 0.01,
            'mean_latency': 0.22,
            'latency_std': 0.07,
            'special_key_freq': 0.20
        }
    }
    
    if user_id not in user_profiles:
        raise ValueError(f"User profile for {user_id} not found")
    
    profile = user_profiles[user_id]
    
    # Common keys
    regular_keys = list('abcdefghijklmnopqrstuvwxyz .,:')
    special_keys = ['shift', 'ctrl', 'alt', 'space', 'backspace', 'enter', 'tab']
    
    all_sessions = []
    
    for session_id in range(num_sessions):
        # Begin the session at a random time
        start_time = datetime.now() - timedelta(days=np.random.randint(1, 10))
        current_time = start_time.timestamp()
        
        session_keystrokes = []
        
        for i in range(keystrokes_per_session):
            # Decide if this is a special key
            is_special = np.random.random() < profile['special_key_freq']
            key = np.random.choice(special_keys if is_special else regular_keys)
            
            # Down time
            down_time = current_time
            
            # Calculate dwell time (hold time)
            dwell_time = max(0.01, np.random.normal(profile['mean_dwell'], profile['dwell_std']))
            
            # Up time
            up_time = down_time + dwell_time
            
            # Add to session
            session_keystrokes.append({
                'key': key,
                'down_time': down_time,
                'up_time': up_time,
                'dwell_time': dwell_time
            })
            
            # Calculate time to next keystroke
            next_latency = max(0.05, np.random.normal(profile['mean_latency'], profile['latency_std']))
            current_time = up_time + next_latency
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(session_keystrokes)
        
        # Calculate additional timing features
        df['next_down_time'] = df['down_time'].shift(-1)
        df['press_press_time'] = df['next_down_time'] - df['down_time']
        df['next_up_time'] = df['up_time'].shift(-1)
        df['release_release_time'] = df['next_up_time'] - df['up_time']
        df['release_press_time'] = df['next_down_time'] - df['up_time']
        
        # Drop the last row which has NaN for next_* columns
        df = df.dropna()
        
        # Calculate session-level metrics
        special_key_counts = {}
        for special_key in special_keys:
            special_key_counts[special_key] = sum(df['key'] == special_key)
        
        session_data = {
            'session_id': f"{user_id}_session_{session_id}",
            'start_time': df['down_time'].min(),
            'end_time': df['up_time'].max(),
            'duration': df['up_time'].max() - df['down_time'].min(),
            'num_keystrokes': len(df),
            'user_id': user_id,
            
            # Timing metrics
            'mean_dwell_time': df['dwell_time'].mean(),
            'std_dwell_time': df['dwell_time'].std(),
            'min_dwell_time': df['dwell_time'].min(),
            'max_dwell_time': df['dwell_time'].max(),
            
            'mean_press_press_time': df['press_press_time'].mean(),
            'std_press_press_time': df['press_press_time'].std(),
            'min_press_press_time': df['press_press_time'].min(),
            'max_press_press_time': df['press_press_time'].max(),
            
            'mean_release_release_time': df['release_release_time'].mean(),
            'std_release_release_time': df['release_release_time'].std(),
            'min_release_release_time': df['release_release_time'].min(),
            'max_release_release_time': df['release_release_time'].max(),
            
            'mean_release_press_time': df['release_press_time'].mean(),
            'std_release_press_time': df['release_press_time'].std(),
            'min_release_press_time': df['release_press_time'].min(),
            'max_release_press_time': df['release_press_time'].max(),
            
            # Special key usage
            'special_key_ratio': sum(df['key'].isin(special_keys)) / len(df),
            **special_key_counts
        }
        
        all_sessions.append(session_data)
    
    return all_sessions

def generate_all_user_data(users, sessions_per_user=5, keystrokes_per_session=100):
    """Generate keystroke data for multiple users"""
    all_data = []
    
    for user_id in users:
        user_sessions = generate_user_keystroke_patterns(
            user_id, 
            num_sessions=sessions_per_user,
            keystrokes_per_session=keystrokes_per_session
        )
        all_data.extend(user_sessions)
    
    return all_data

def save_combined_data(data, output_file):
    """Save the combined user data to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump({'session_features': data}, f, indent=2)
    
    print(f"Saved {len(data)} sessions to {output_file}")

def main():
    # Define users to generate data for
    users = ['user1', 'user2', 'user3', 'user4', 'user5']
    
    # Generate synthetic data
    all_data = generate_all_user_data(
        users,
        sessions_per_user=10,  # 10 sessions per user
        keystrokes_per_session=100  # 100 keystrokes per session
    )
    
    # Save the combined data
    save_combined_data(all_data, "combined_users_data.json")
    
    # Generate a test session for authentication
    test_user = np.random.choice(users)
    test_session = generate_user_keystroke_patterns(
        test_user,
        num_sessions=1,
        keystrokes_per_session=50
    )
    
    save_combined_data(test_session, "test_session.json")
    print(f"Generated test session for {test_user}")

if __name__ == "__main__":
    main()
