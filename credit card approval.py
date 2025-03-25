import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class CreditCardApprovalSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def preprocess_data(self, df):
        df_processed = df.copy()
        
        numerical_columns = df_processed.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        for col in numerical_columns:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
      
        for col in categorical_columns:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
            
        # Encode categorical variables
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            
        return df_processed
    
    def prepare_features(self, df_processed, target_column=None):
       
        if target_column and target_column in df_processed.columns:
            X = df_processed.drop(columns=[target_column])
            y = df_processed[target_column]
        else:
            X = df_processed
            y = None
            
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        return X_scaled, y
    
    def train(self, X, y):
        # Calculate class weights
        unique_classes = np.unique(y)
        class_weights = dict(zip(unique_classes, 
                               [1 / (np.sum(y == c) / len(y)) for c in unique_classes]))
        
        # Initialize and train Random Forest model with class weights
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=class_weights,  # Handle imbalanced classes
            random_state=42
        )
        self.model.fit(X, y)
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        print(f"Model Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(report)
        
        # Feature importance plot
        self.plot_feature_importance()
        
    def plot_feature_importance(self):
        importances = self.model.feature_importances_
        features_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=features_df, x='importance', y='feature')
        plt.title('Feature Importance in Credit Card Approval Prediction')
        plt.show()

# Example usage:
def main():
    # Load your dataset
    # Replace this with your actual dataset loading code
    # data = pd.read_csv('credit_card_approval_dataset.csv')
    
    # For demonstration, creating a sample dataset
    data = pd.DataFrame({
        'income': np.random.randint(20000, 200000, 1000),
        'age': np.random.randint(18, 70, 1000),
        'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], 1000),
        'credit_score': np.random.randint(300, 850, 1000),
        'approval_status': np.random.choice([0, 1], 1000, p=[0.7, 0.3])  # Target variable
    })
    
    # Initialize the system
    approval_system = CreditCardApprovalSystem()
    
    # Preprocess the data
    processed_data = approval_system.preprocess_data(data)
    
    # Prepare features and target
    X, y = approval_system.prepare_features(processed_data, target_column='approval_status')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    approval_system.train(X_train, y_train)
    
    # Evaluate the model
    approval_system.evaluate(X_test, y_test)
    
    # Example prediction for a new application
    new_application = pd.DataFrame({
        'income': [75000],
        'age': [35],
        'employment_status': ['Employed'],
        'credit_score': [720]
    })
    
    # Preprocess the new application
    processed_application = approval_system.preprocess_data(new_application)
    X_new, _ = approval_system.prepare_features(processed_application)
    
    # Make prediction
    prediction = approval_system.predict(X_new)
    print(f"\nPrediction for new application: {'Approved' if prediction[0] == 1 else 'Denied'}")

if __name__ == "__main__":
    main()
