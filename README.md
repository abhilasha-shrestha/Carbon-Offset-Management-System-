# Carbon Offset ML Model

An intelligent machine learning system that recommends optimal tree species and quantities for carbon offsetting based on industrial emissions, budget, and geographical constraints.

## Features

- **Hybrid Recommendation Engine**: Combines rule-based logic with ML (Random Forest + Gradient Boosting) for optimal suggestions.
- **Dynamic Learning**: Generates synthetic training data based on real-world constraints.
- **Unity Integration**: Exports data in a format ready for visualization in Unity.
- **Detailed Metrics**: Calculates carbon offset, costs, land requirements, and biodiversity scores.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Train the Model

Train the ML models and save them to `model_weights/`:

```bash
python train.py --samples 5000
```

### 2. Make Predictions

Generate recommendations for a specific scenario:

```bash
python predict.py \
  --emissions 500000 \
  --years 15 \
  --budget 50000 \
  --space 20000 \
  --use-ml
```

**Arguments:**
- `--emissions`: Annual CO2 emissions in kg (required)
- `--years`: Project duration in years (default: 10)
- `--budget`: Maximum budget in USD
- `--space`: Available land in square meters
- `--use-ml`: Use the trained ML model instead of simple rules
- `--unity-format`: Output results in a format suitable for Unity integration

## Project Structure

- `carbon_models.py`: Core logic for the ML model and Carbon Calculator.
- `train.py`: Script to generate data and train the model.
- `predict.py`: CLI tool for making predictions.
- `tree_database.py`: Database of tree species and their carbon sequestration rates.
- `test_unity_connection.py`: Test script for verifying API connections.
