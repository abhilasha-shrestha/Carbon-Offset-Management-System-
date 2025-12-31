"""
Training script for Carbon Offset ML Model
"""

import argparse
import pandas as pd
from carbon_models import create_and_train_model, CarbonOffsetMLModel
import joblib

def main():
    parser = argparse.ArgumentParser(description='Train Carbon Offset ML Model')
    parser.add_argument('--samples', type=int, default=3000, 
                       help='Number of training samples')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion')
    parser.add_argument('--save_dir', type=str, default='model_weights',
                       help='Directory to save model')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to existing training data (optional)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CARBON OFFSET ML MODEL TRAINING")
    print("="*60)
    
    # Load existing data or generate new
    if args.data_path:
        print(f"Loading training data from {args.data_path}")
        training_data = pd.read_csv(args.data_path)
    else:
        training_data = None
    
    # Create and train model
    model = CarbonOffsetMLModel()
    
    if training_data is not None:
        accuracy, r2, rmse = model.train(
            training_data=training_data,
            test_size=args.test_size
        )
    else:
        # Generate and train
        accuracy, r2, rmse = model.train(
            training_data=None,
            test_size=args.test_size
        )
    
    # Save model
    model.save_model(args.save_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved to: {args.save_dir}/carbon_model.pkl")
    print(f"Species Accuracy: {accuracy:.2%}")
    print(f"Count R² Score: {r2:.4f}")
    print(f"Count RMSE: {rmse:.2f} trees")
    
    # Test with example
    print("\n" + "="*60)
    print("TEST PREDICTION")
    print("="*60)
    
    test_case = {
        'annual_emissions_kg': 1000000,
        'duration_years': 10,
        'budget_usd': 50000,
        'space_m2': 20000,
        'industry': 'manufacturing',
        'climate_zone': 'temperate'
    }
    
    result = model.predict(**test_case, use_ml=True)
    
    if 'recommendation' in result:
        rec = result['recommendation']
        print(f"\nTest Prediction Successful!")
        print(f"Species: {rec['common_name']}")
        print(f"Trees: {rec['trees_needed']:,}")
        print(f"Cost: ${rec['total_cost_usd']:,.2f}")
        print(f"Carbon Offset: {rec['total_carbon_offset_kg']/1000:,.1f} tons")
    else:
        print(f"\nTest Prediction Failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()