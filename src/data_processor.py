"""
Data processing functions for logistics cost analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict

class LogisticsDataProcessor:
    """Process and clean logistics data for analysis."""
    
    def __init__(self, high_cost_threshold: float = 1000):
        self.high_cost_threshold = high_cost_threshold
        self.cleaning_report = {}
    
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """Load data and perform basic validation."""
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Data loaded successfully: {df.shape[0]} records")
            
            # Basic validation
            required_columns = ['shipment_id', 'ship_date', 'invoice_amount', 'billable_weight']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def calculate_cost_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cost per kg and identify anomalies."""
        df = df.copy()
        
        # Calculate cost per kg
        df['cost_per_kg'] = df['invoice_amount'] / df['billable_weight']
        
        # Identify records needing review
        df['needs_review'] = df['cost_per_kg'] > self.high_cost_threshold
        
        self.cleaning_report['total_records'] = len(df)
        self.cleaning_report['anomalies_detected'] = df['needs_review'].sum()
        
        print(f"📊 Cost metrics calculated: {df['needs_review'].sum()} anomalies detected")
        
        return df
    
    def handle_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle anomalous records and create cleaned weight column."""
        df = df.copy()
        
        # Create cleaned weight (0 for service fees, actual weight for freight)
        df['cleaned_billable_weight'] = np.where(
            df['needs_review'], 
            0,  # Service fees get 0 weight
            df['billable_weight']  # Normal freight keeps actual weight
        )
        
        # Categorize charge types
        def categorize_charge(row):
            if row['needs_review']:
                return 'Fixed Fee Service'
            elif row['billable_weight'] < 10 and row['invoice_amount'] > 1000:
                return 'Minimum Charge'
            else:
                return 'Standard Freight'
        
        df['charge_type'] = df.apply(categorize_charge, axis=1)
        
        # Calculate cleaned cost per kg for standard freight
        standard_freight_mask = df['charge_type'] == 'Standard Freight'
        df.loc[standard_freight_mask, 'cleaned_cost_per_kg'] = (
            df.loc[standard_freight_mask, 'invoice_amount'] / 
            df.loc[standard_freight_mask, 'cleaned_billable_weight']
        )
        
        self.cleaning_report['charge_type_distribution'] = df['charge_type'].value_counts().to_dict()
        
        print("✅ Anomalies handled and data cleaned")
        
        return df
    
    def get_cleaning_report(self) -> Dict:
        """Get detailed report of data cleaning operations."""
        return self.cleaning_report
    
    def process_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete data processing pipeline."""
        print("🚀 Starting data processing pipeline...")
        
        # Load data
        df = self.load_and_validate_data(file_path)
        
        # Calculate cost metrics
        df = self.calculate_cost_metrics(df)
        
        # Handle anomalies
        df = self.handle_anomalies(df)
        
        # Create standard freight dataset
        standard_freight = df[df['charge_type'] == 'Standard Freight'].copy()
        
        print("🎉 Data processing completed successfully!")
        print(f"📊 Final datasets: Full={len(df)}, Standard Freight={len(standard_freight)}")
        
        return df, standard_freight

# Example usage
if __name__ == "__main__":
    processor = LogisticsDataProcessor()
    full_data, standard_freight = processor.process_data("../data/raw/sample_data.csv")
    print(processor.get_cleaning_report())