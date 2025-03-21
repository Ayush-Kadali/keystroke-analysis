import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class KeystrokeDynamicsIdentifier:
    """
    A class for identifying users based on keystroke dynamics.
    This system processes keystroke data, trains multiple models, 
    and implements a simple expert system for authentication decisions.
    """
    
    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.scaler = StandardScaler()
        self.trained = False
        self.expert_system_threshold = 0.7  # Confidence threshold for expert system
    
    def load_data(self, file_path):
        """Load processed keystroke data from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract session features
        sessions = pd.DataFrame(data['session_features'])
        return sessions
    
    def process_raw_data(self, raw_file_path, processed_file_path, user_id):
        """Process raw keystroke data and label it with user ID"""
        # This assumes you've already created the process_keystroke_data function
        # Import and use that function here
        from process_keystrokes import process_keystroke_data
        
        df, session_df = process_keystroke_data(raw_file_path, processed_file_path)
        
        # Add user ID to each session
        session_df['user_id'] = user_id
        
        return session_df
    
    def combine_user_data(self, dataframes):
        """Combine data from multiple users"""
        return pd.concat(dataframes, ignore_index=True)
    
    def prepare_features(self, data):
        """Prepare features for training the models"""
        # Define feature columns - timing features and special key usage
        timing_cols = [col for col in data.columns if any(x in col for x in 
                     ['dwell_time', 'press_press_time', 'release_release_time', 'release_press_time'])]
        
        special_key_cols = ['shift', 'ctrl', 'alt', 'space', 'backspace', 'enter', 'tab', 'special_key_ratio']
        
        # Filter to only include columns that exist in the data
        timing_cols = [col for col in timing_cols if col in data.columns]
        special_key_cols = [col for col in special_key_cols if col in data.columns]
        
        self.feature_columns = timing_cols + special_key_cols
        
        # Check if we have enough features
        if len(self.feature_columns) < 3:
            raise ValueError("Not enough features available in the data")
            
        # Prepare X (features) and y (target)
        X = data[self.feature_columns].copy()
        y = data['user_id']
        
        # Handle NaN values
        X = X.fillna(0)
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple models for user identification"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Fit the scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to train
        models_to_train = {
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'naive_bayes': GaussianNB()
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in models_to_train.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate on test set
            y_pred = model.predict(X_test_scaled)
            accuracy = model.score(X_test_scaled, y_test)
            
            # Store trained model
            self.models[name] = model
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
        
        self.trained = True
        return results, X_test, y_test
    
    def identify_user(self, session_data):
        """Identify a user based on a single typing session"""
        if not self.trained:
            raise ValueError("Models have not been trained yet.")
        
        # Prepare features
        features = session_data[self.feature_columns].fillna(0)
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            # Get predicted class and probabilities
            pred_class = model.predict(features_scaled)[0]
            
            # Get probability scores if the model supports it
            try:
                pred_proba = model.predict_proba(features_scaled)[0]
                max_proba = np.max(pred_proba)
            except:
                max_proba = 1.0  # Default if probabilities not available
                
            predictions[name] = {
                'predicted_user': pred_class,
                'confidence': max_proba
            }
        
        return predictions
    
    def expert_system_authenticate(self, predictions):
        """
        Expert system to make authentication decisions based on model predictions.
        
        Rules:
        1. If all models agree on the user, and average confidence > threshold, authenticate
        2. If random_forest and neural_network agree with high confidence, authenticate
        3. If SVM has very high confidence (>0.9), authenticate
        4. Otherwise, deny authentication
        """
        # Get all predicted users
        predicted_users = [pred['predicted_user'] for pred in predictions.values()]
        
        # Check if all models predict the same user
        if len(set(predicted_users)) == 1:
            # All models agree on the user
            avg_confidence = np.mean([pred['confidence'] for pred in predictions.values()])
            if avg_confidence > self.expert_system_threshold:
                return {
                    'authenticated': True,
                    'user': predicted_users[0],
                    'confidence': avg_confidence,
                    'rule': 'unanimous_agreement'
                }
        
        # Check if random forest and neural network agree
        if ('random_forest' in predictions and 'neural_network' in predictions and
            predictions['random_forest']['predicted_user'] == predictions['neural_network']['predicted_user']):
            
            rf_conf = predictions['random_forest']['confidence']
            nn_conf = predictions['neural_network']['confidence']
            avg_conf = (rf_conf + nn_conf) / 2
            
            if avg_conf > self.expert_system_threshold + 0.1:  # Higher threshold for this rule
                return {
                    'authenticated': True,
                    'user': predictions['random_forest']['predicted_user'],
                    'confidence': avg_conf,
                    'rule': 'rf_nn_agreement'
                }
        
        # Check if SVM has very high confidence
        if 'svm' in predictions and predictions['svm']['confidence'] > 0.9:
            return {
                'authenticated': True,
                'user': predictions['svm']['predicted_user'],
                'confidence': predictions['svm']['confidence'],
                'rule': 'svm_high_confidence'
            }
        
        # Default: not authenticated
        return {
            'authenticated': False,
            'confidence': max([pred['confidence'] for pred in predictions.values()]),
            'rule': 'insufficient_confidence'
        }
    
    def evaluate_models_cross_val(self, X, y, cv=5):
        """Evaluate models using cross-validation"""
        cv_results = {}
        
        for name, model in self.models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            # Perform cross-validation
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
            
            cv_results[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'scores': scores.tolist()
            }
        
        return cv_results
    
    def get_feature_importance(self):
        """Get feature importance for models that support it"""
        importance_data = {}
        
        for name, model in self.models.items():
            # Check if model has feature_importances_ attribute
            if hasattr(model, 'feature_importances_'):
                # Create a dataframe of feature importances
                importance = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_data[name] = importance.to_dict('records')
        
        return importance_data

def main():
    """Example usage of the KeystrokeDynamicsIdentifier class"""
    # Initialize the identifier
    identifier = KeystrokeDynamicsIdentifier()
    
    # For a real application, you would process raw data for each user like this:
    # user1_data = identifier.process_raw_data("user1_events.txt", "user1_processed.json", "user1")
    # user2_data = identifier.process_raw_data("user2_events.txt", "user2_processed.json", "user2")
    # user3_data = identifier.process_raw_data("user3_events.txt", "user3_processed.json", "user3")
    # all_data = identifier.combine_user_data([user1_data, user2_data, user3_data])
    
    # For demonstration, we'll load pre-processed data:
    all_data = identifier.load_data("combined_users_data.json")
    
    # Prepare features and train models
    X, y = identifier.prepare_features(all_data)
    results, X_test, y_test = identifier.train_models(X, y)
    
    # Print model performance
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()} PERFORMANCE:")
        print(f"Accuracy: {model_results['accuracy']:.4f}")
        
        report = model_results['report']
        print("\nClassification Report:")
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"  Class {class_name}:")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall: {metrics['recall']:.4f}")
                print(f"    F1-score: {metrics['f1-score']:.4f}")
    
    # Perform cross-validation
    cv_results = identifier.evaluate_models_cross_val(X, y)
    print("\nCROSS-VALIDATION RESULTS:")
    for model_name, cv_result in cv_results.items():
        print(f"{model_name}: Mean accuracy = {cv_result['mean_accuracy']:.4f} (±{cv_result['std_accuracy']:.4f})")
    
    # Example of using the trained models for user identification
    # Get a sample session for testing
    sample_session = X_test.iloc[0].to_frame().T
    sample_session['user_id'] = y_test.iloc[0]  # Add actual user ID for verification
    
    # Identify the user
    predictions = identifier.identify_user(sample_session)
    
    # Apply expert system for authentication decision
    auth_result = identifier.expert_system_authenticate(predictions)
    
    print("\nAUTHENTICATION TEST:")
    print(f"Actual user: {sample_session['user_id'].iloc[0]}")
    print(f"Authentication result: {auth_result}")
    
    # Get feature importance
    feature_importance = identifier.get_feature_importance()
    if feature_importance:
        print("\nFEATURE IMPORTANCE:")
        for model_name, importance in feature_importance.items():
            print(f"\n{model_name.upper()}:")
            for feature in importance[:5]:  # Show top 5 features
                print(f"  {feature['feature']}: {feature['importance']:.4f}")

if __name__ == "__main__":
    main()
