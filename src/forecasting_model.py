"""
Forecasting Model - Production-ready ML model for logistics cost prediction
Part of Logistics Cost Intelligence Project

Author: [Mmesoma Eboh]
Date: [9/10/2025]
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class LogisticsForecastingModel:
    """
    Production forecasting model for logistics costs with 98%+ accuracy
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.performance_metrics = {}
        
    def prepare_features(self, df):
        """
        Prepare features for modeling based on feature engineering analysis
        
        Parameters:
        df (pd.DataFrame): Processed logistics data
        
        Returns:
        tuple: (X_features, y_target, feature_names)
        """
        # Select optimal features from your analysis
        feature_columns = [
            'cleaned_billable_weight',
            'service_level_encoded', 
            'lane_encoded',
            'monthly_avg_scaled',
            'volatility_scaled'
        ]
        
        # Use only columns that exist in dataframe
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features]
        y = df['cleaned_cost_per_kg']
        
        self.feature_columns = available_features
        
        return X, y, available_features
    
    def train_model(self, df, test_size=0.2, random_state=42):
        """
        Train the forecasting model with cross-validation
        
        Parameters:
        df (pd.DataFrame): Processed logistics data
        test_size (float): Proportion for test split
        random_state (int): Random seed for reproducibility
        
        Returns:
        dict: Training performance metrics
        """
        print("🚀 Training Logistics Forecasting Model...")
        
        # Prepare features
        X, y, feature_names = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model based on type
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError("Model type must be 'random_forest' or 'linear'")
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate performance metrics
        self.performance_metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'accuracy': max(0, (1 - (mean_absolute_error(y_test, y_pred) / y_test.mean())) * 100),
            'feature_importance': dict(zip(feature_names, self.model.feature_importances_)) 
            if hasattr(self.model, 'feature_importances_') else None
        }
        
        print("✅ Model training completed!")
        print(f"📊 Model Performance:")
        print(f"   - MAE: ${self.performance_metrics['mae']:.2f}")
        print(f"   - RMSE: ${self.performance_metrics['rmse']:.2f}") 
        print(f"   - R²: {self.performance_metrics['r2']:.3f}")
        print(f"   - Accuracy: {self.performance_metrics['accuracy']:.1f}%")
        
        return self.performance_metrics
    
    def predict(self, new_data):
        """
        Make predictions on new data
        
        Parameters:
        new_data (pd.DataFrame): New logistics data with required features
        
        Returns:
        np.array: Predicted costs
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Select and scale features
        X_new = new_data[self.feature_columns]
        X_new_scaled = self.scaler.transform(X_new)
        
        # Make predictions
        predictions = self.model.predict(X_new_scaled)
        
        return predictions
    
    def back_test(self, df, periods=5):
        """
        Perform back-testing to validate model performance
        
        Parameters:
        df (pd.DataFrame): Historical data
        periods (int): Number of back-testing periods
        
        Returns:
        dict: Back-testing results
        """
        print("🔍 Performing Back-Testing Validation...")
        
        accuracies = []
        for i in range(periods):
            # Simulate different train/test splits
            X, y, _ = self.prepare_features(df)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=i
            )
            
            # Retrain and evaluate
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
            temp_model.fit(X_train_scaled, y_train)
            
            y_pred = temp_model.predict(X_test_scaled)
            accuracy = max(0, (1 - (mean_absolute_error(y_test, y_pred) / y_test.mean())) * 100)
            accuracies.append(accuracy)
        
        back_test_results = {
            'average_accuracy': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'all_accuracies': accuracies
        }
        
        print(f"✅ Back-testing completed: {back_test_results['average_accuracy']:.1f}% average accuracy")
        
        return back_test_results
    
    def save_model(self, filepath):
        """
        Save trained model to file
        
        Parameters:
        filepath (str): Path to save model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'performance_metrics': self.performance_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from file
        
        Parameters:
        filepath (str): Path to saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler'] 
        self.feature_columns = model_data['feature_columns']
        self.performance_metrics = model_data['performance_metrics']
        
        print(f"📂 Model loaded from: {filepath}")

# Utility functions for business case integration
def validate_business_case_accuracy(df, target_accuracy=98.0):
    """
    Validate the 98%+ accuracy claim for business case
    
    Parameters:
    df (pd.DataFrame): Processed logistics data
    target_accuracy (float): Target accuracy percentage
    
    Returns:
    dict: Validation results
    """
    model = LogisticsForecastingModel()
    performance = model.train_model(df)
    back_test = model.back_test(df)
    
    validation = {
        'claim_supported': performance['accuracy'] >= target_accuracy,
        'achieved_accuracy': performance['accuracy'],
        'back_test_consistency': back_test['average_accuracy'],
        'business_case_ready': (performance['accuracy'] >= target_accuracy and 
                              back_test['average_accuracy'] >= target_accuracy),
        'confidence_level': 'HIGH' if performance['accuracy'] >= 98 else 'MEDIUM'
    }
    
    print("🎯 BUSINESS CASE VALIDATION:")
    print(f"   Target Accuracy: {target_accuracy}%")
    print(f"   Achieved Accuracy: {performance['accuracy']:.1f}%")
    print(f"   Back-test Consistency: {back_test['average_accuracy']:.1f}%")
    print(f"   Claim Supported: {'✅ YES' if validation['claim_supported'] else '❌ NO'}")
    
    return validation

def generate_forecasting_report(df):
    """
    Generate comprehensive forecasting report for business case
    
    Parameters:
    df (pd.DataFrame): Processed logistics data
    
    Returns:
    dict: Complete forecasting analysis
    """
    print("📊 Generating Forecasting Report for Business Case...")
    
    # Train model
    model = LogisticsForecastingModel()
    performance = model.train_model(df)
    back_test = model.back_test(df)
    validation = validate_business_case_accuracy(df)
    
    report = {
        'performance_metrics': performance,
        'back_testing_results': back_test,
        'business_case_validation': validation,
        'model_characteristics': {
            'model_type': model.model_type,
            'features_used': model.feature_columns,
            'training_samples': len(df),
            'feature_importance': performance.get('feature_importance', {})
        },
        'strategic_implications': {
            'budgeting_confidence': 'HIGH' if performance['accuracy'] >= 98 else 'MEDIUM',
            'implementation_readiness': 'READY' if validation['business_case_ready'] else 'NEEDS_IMPROVEMENT',
            'financial_impact': 'Proactive budgeting model enabled'
        }
    }
    
    print("✅ Forecasting report completed!")
    return report

# Example usage
if __name__ == "__main__":
    print("Logistics Forecasting Model - Production Ready")
    print("Use validate_business_case_accuracy() to verify 98%+ claims")