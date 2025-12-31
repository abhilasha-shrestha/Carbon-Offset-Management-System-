"""
Prediction script for Carbon Offset ML Model
"""

import argparse
import json
from carbon_models import load_trained_model

def main():
    parser = argparse.ArgumentParser(description='Make predictions with Carbon Offset ML Model')
    parser.add_argument('--model', type=str, default='model_weights/carbon_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--emissions', type=float, required=True,
                       help='Annual CO2 emissions in kg')
    parser.add_argument('--years', type=int, default=10,
                       help='Project duration in years')
    parser.add_argument('--budget', type=float, help='Budget in USD')
    parser.add_argument('--space', type=float, help='Available space in m²')
    parser.add_argument('--industry', type=str, default='manufacturing',
                       help='Industry type')
    parser.add_argument('--climate', type=str, help='Climate zone')
    parser.add_argument('--use-ml', action='store_true',
                       help='Use ML model (default: rule-based)')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--unity-format', action='store_true',
                       help='Output in Unity format')
    
    args = parser.parse_args()
    
    # print("Loading model...")
    model = load_trained_model(args.model)
    
    # print("Making prediction...")
    result = model.predict(
        annual_emissions_kg=args.emissions,
        duration_years=args.years,
        budget_usd=args.budget,
        space_m2=args.space,
        industry=args.industry,
        climate_zone=args.climate,
        use_ml=args.use_ml
    )
    
    # Output results
    if args.unity_format:
        output = model.export_for_unity(result)
    else:
        output = result
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")
    # else:
    #     print(json.dumps(output, indent=2))
    
    # Print summary to console
    if 'recommendation' in result:
        rec = result['recommendation']
        # print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(f"Species: {rec.get('common_name', 'N/A')}")
        print(f"Trees Needed: {rec.get('trees_needed', 0):,}")
        print(f"Total Cost: ${rec.get('total_cost_usd', 0):,.2f}")
        print(f"Carbon Offset: {rec.get('total_carbon_offset_kg', 0)/1000:,.1f} tons")
        print(f"Land Required: {rec.get('land_required_m2', 0):,.0f} m²")

if __name__ == "__main__":
    main()