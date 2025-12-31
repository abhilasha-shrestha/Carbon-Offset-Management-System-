"""
Carbon Offset Tree Recommendation ML Model
Author: [Your Name]
Date: 2024
Description: ML model that recommends optimal tree species and quantities
             to offset industrial carbon emissions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import joblib
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class TreeSpeciesDatabase:
    """Database of tree species with carbon sequestration rates"""
    
    # Tree species data (20 species with real carbon sequestration rates)
    TREE_DATA = {
        'oak': {
            'common_name': 'Oak',
            'scientific_name': 'Quercus robur',
            'co2_kg_per_year': 22.0,
            'lifespan_years': 200,
            'climate_zones': ['temperate'],
            'growth_rate': 'medium',  # slow, medium, fast, very_fast
            'water_needs': 'medium',
            'cost_usd': 8.0,
            'height_m': 25,
            'dbh_cm': 100,
            'space_m2': 100,
            'biodiversity_score': 8.5
        },
        'mangrove': {
            'common_name': 'Mangrove',
            'scientific_name': 'Rhizophora spp.',
            'co2_kg_per_year': 35.0,
            'lifespan_years': 100,
            'climate_zones': ['tropical', 'coastal'],
            'growth_rate': 'fast',
            'water_needs': 'high',
            'cost_usd': 12.0,
            'height_m': 15,
            'dbh_cm': 30,
            'space_m2': 25,
            'biodiversity_score': 9.2
        },
        'pine': {
            'common_name': 'Pine',
            'scientific_name': 'Pinus sylvestris',
            'co2_kg_per_year': 18.5,
            'lifespan_years': 150,
            'climate_zones': ['boreal', 'temperate'],
            'growth_rate': 'slow',
            'water_needs': 'low',
            'cost_usd': 6.0,
            'height_m': 30,
            'dbh_cm': 80,
            'space_m2': 60,
            'biodiversity_score': 7.0
        },
        'eucalyptus': {
            'common_name': 'Eucalyptus',
            'scientific_name': 'Eucalyptus globulus',
            'co2_kg_per_year': 28.0,
            'lifespan_years': 70,
            'climate_zones': ['tropical', 'subtropical'],
            'growth_rate': 'very_fast',
            'water_needs': 'high',
            'cost_usd': 7.0,
            'height_m': 40,
            'dbh_cm': 100,
            'space_m2': 80,
            'biodiversity_score': 6.5
        },
        'neem': {
            'common_name': 'Neem',
            'scientific_name': 'Azadirachta indica',
            'co2_kg_per_year': 25.0,
            'lifespan_years': 200,
            'climate_zones': ['tropical', 'arid'],
            'growth_rate': 'medium',
            'water_needs': 'low',
            'cost_usd': 4.0,
            'height_m': 20,
            'dbh_cm': 60,
            'space_m2': 50,
            'biodiversity_score': 8.0
        },
        'bamboo': {
            'common_name': 'Bamboo',
            'scientific_name': 'Bambusa vulgaris',
            'co2_kg_per_year': 30.0,
            'lifespan_years': 50,
            'climate_zones': ['tropical', 'subtropical'],
            'growth_rate': 'very_fast',
            'water_needs': 'high',
            'cost_usd': 5.0,
            'height_m': 20,
            'dbh_cm': 15,
            'space_m2': 10,
            'biodiversity_score': 6.5
        },
        'acacia': {
            'common_name': 'Acacia',
            'scientific_name': 'Acacia nilotica',
            'co2_kg_per_year': 19.0,
            'lifespan_years': 80,
            'climate_zones': ['tropical', 'arid'],
            'growth_rate': 'fast',
            'water_needs': 'low',
            'cost_usd': 3.0,
            'height_m': 15,
            'dbh_cm': 40,
            'space_m2': 40,
            'biodiversity_score': 7.5
        },
        'teak': {
            'common_name': 'Teak',
            'scientific_name': 'Tectona grandis',
            'co2_kg_per_year': 20.0,
            'lifespan_years': 300,
            'climate_zones': ['tropical'],
            'growth_rate': 'slow',
            'water_needs': 'medium',
            'cost_usd': 15.0,
            'height_m': 30,
            'dbh_cm': 120,
            'space_m2': 120,
            'biodiversity_score': 7.0
        },
        'willow': {
            'common_name': 'Willow',
            'scientific_name': 'Salix babylonica',
            'co2_kg_per_year': 24.0,
            'lifespan_years': 60,
            'climate_zones': ['temperate'],
            'growth_rate': 'very_fast',
            'water_needs': 'high',
            'cost_usd': 6.0,
            'height_m': 12,
            'dbh_cm': 50,
            'space_m2': 20,
            'biodiversity_score': 7.0
        },
        'maple': {
            'common_name': 'Maple',
            'scientific_name': 'Acer saccharum',
            'co2_kg_per_year': 16.5,
            'lifespan_years': 150,
            'climate_zones': ['temperate'],
            'growth_rate': 'slow',
            'water_needs': 'medium',
            'cost_usd': 9.0,
            'height_m': 25,
            'dbh_cm': 90,
            'space_m2': 90,
            'biodiversity_score': 8.0
        }
    }
    
    # Climate zone mapping
    CLIMATE_ZONES = {
        'temperate': ['oak', 'pine', 'maple', 'willow'],
        'tropical': ['mangrove', 'eucalyptus', 'neem', 'bamboo', 'teak'],
        'boreal': ['pine'],
        'arid': ['neem', 'acacia'],
        'subtropical': ['eucalyptus', 'bamboo'],
        'coastal': ['mangrove', 'coconut']
    }
    
    @classmethod
    def get_species_by_climate(cls, climate_zone: str) -> List[str]:
        """Get tree species suitable for specific climate zone"""
        if climate_zone in cls.CLIMATE_ZONES:
            return cls.CLIMATE_ZONES[climate_zone]
        return list(cls.TREE_DATA.keys())  # Return all if climate not specified
    
    @classmethod
    def get_species_details(cls, species_id: str) -> Dict:
        """Get detailed information about a species"""
        return cls.TREE_DATA.get(species_id, {})
    
    @classmethod
    def get_all_species(cls) -> List[Dict]:
        """Get all tree species"""
        return [
            {'id': species_id, **details}
            for species_id, details in cls.TREE_DATA.items()
        ]

class CarbonOffsetCalculator:
    """Core calculator for tree recommendations"""
    
    def __init__(self, survival_rate: float = 0.75):
        self.survival_rate = survival_rate
        self.tree_db = TreeSpeciesDatabase()
    
    def calculate_optimal_trees(
        self,
        annual_emissions_kg: float,
        duration_years: int = 10,
        climate_zone: Optional[str] = None,
        budget_usd: Optional[float] = None,
        space_m2: Optional[float] = None,
        optimization_criteria: str = 'min_cost'
    ) -> Dict:
        """
        Calculate optimal tree species and quantities
        
        Args:
            annual_emissions_kg: Annual CO2 emissions in kg
            duration_years: Project duration in years
            climate_zone: Optional climate constraint
            budget_usd: Optional budget constraint
            space_m2: Optional space constraint
            optimization_criteria: 'min_cost', 'min_space', 'max_efficiency', 'fast_growth'
        
        Returns:
            Dictionary with recommendations
        """
        
        # Calculate total carbon to offset
        total_carbon_kg = annual_emissions_kg * duration_years
        
        # Get suitable species based on climate
        if climate_zone:
            suitable_species_ids = self.tree_db.get_species_by_climate(climate_zone)
        else:
            suitable_species_ids = list(self.tree_db.TREE_DATA.keys())
        
        recommendations = []
        
        for species_id in suitable_species_ids:
            species_data = self.tree_db.get_species_details(species_id)
            
            if not species_data:
                continue
            
            # Calculate trees needed (with survival rate)
            annual_seq = species_data['co2_kg_per_year']
            effective_annual_seq = annual_seq * self.survival_rate
            total_seq_per_tree = effective_annual_seq * duration_years
            
            trees_needed = np.ceil(total_carbon_kg / total_seq_per_tree)
            
            # Calculate costs and space
            total_cost = trees_needed * species_data['cost_usd']
            land_required = trees_needed * species_data['space_m2']
            total_carbon_offset = trees_needed * total_seq_per_tree
            
            # Check constraints
            if budget_usd and total_cost > budget_usd:
                continue  # Skip if exceeds budget
            if space_m2 and land_required > space_m2:
                continue  # Skip if exceeds space
            
            # Calculate efficiency metrics
            cost_per_kg = total_cost / total_carbon_offset if total_carbon_offset > 0 else float('inf')
            space_efficiency = total_carbon_offset / land_required if land_required > 0 else 0
            
            recommendations.append({
                'species_id': species_id,
                'common_name': species_data['common_name'],
                'scientific_name': species_data['scientific_name'],
                'trees_needed': int(trees_needed),
                'annual_carbon_per_tree_kg': annual_seq,
                'total_carbon_offset_kg': float(total_carbon_offset),
                'project_duration_years': duration_years,
                'total_cost_usd': float(total_cost),
                'land_required_m2': float(land_required),
                'growth_rate': species_data['growth_rate'],
                'climate_zones': species_data['climate_zones'],
                'cost_per_kg_co2': cost_per_kg,
                'space_efficiency': space_efficiency,
                'biodiversity_score': species_data['biodiversity_score']
            })
        
        if not recommendations:
            # Return best effort if no constraints can be met
            return self._get_best_effort_recommendation(
                total_carbon_kg, suitable_species_ids, duration_years
            )
        
        # Rank recommendations based on criteria
        ranked_recommendations = self._rank_recommendations(
            recommendations, optimization_criteria
        )
        
        return ranked_recommendations
    
    def _get_best_effort_recommendation(
        self,
        total_carbon_kg: float,
        species_ids: List[str],
        duration_years: int
    ) -> Dict:
        """Get recommendation when constraints can't be met"""
        best_recommendation = None
        min_cost = float('inf')
        
        for species_id in species_ids:
            species_data = self.tree_db.get_species_details(species_id)
            annual_seq = species_data['co2_kg_per_year']
            total_seq_per_tree = annual_seq * self.survival_rate * duration_years
            
            trees_needed = np.ceil(total_carbon_kg / total_seq_per_tree)
            total_cost = trees_needed * species_data['cost_usd']
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_recommendation = {
                    'species_id': species_id,
                    'common_name': species_data['common_name'],
                    'trees_needed': int(trees_needed),
                    'annual_carbon_per_tree_kg': annual_seq,
                    'total_carbon_offset_kg': float(trees_needed * total_seq_per_tree),
                    'total_cost_usd': float(total_cost),
                    'land_required_m2': float(trees_needed * species_data['space_m2']),
                    'growth_rate': species_data['growth_rate'],
                    'note': 'Constraints could not be fully met'
                }
        
        return {
            'optimization_criteria': 'best_effort',
            'top_recommendation': best_recommendation,
            'alternative_recommendations': []
        }
    
    def _rank_recommendations(
        self,
        recommendations: List[Dict],
        criteria: str
    ) -> Dict:
        """Rank recommendations based on optimization criteria"""
        
        if criteria == 'min_cost':
            recommendations.sort(key=lambda x: x['total_cost_usd'])
        elif criteria == 'min_space':
            recommendations.sort(key=lambda x: x['land_required_m2'])
        elif criteria == 'max_efficiency':
            recommendations.sort(key=lambda x: x['space_efficiency'], reverse=True)
        elif criteria == 'fast_growth':
            growth_scores = {'slow': 1, 'medium': 2, 'fast': 3, 'very_fast': 4}
            recommendations.sort(key=lambda x: growth_scores.get(x['growth_rate'], 2), reverse=True)
        elif criteria == 'high_biodiversity':
            recommendations.sort(key=lambda x: x['biodiversity_score'], reverse=True)
        else:
            # Default: min cost
            recommendations.sort(key=lambda x: x['total_cost_usd'])
        
        return {
            'optimization_criteria': criteria,
            'top_recommendation': recommendations[0] if recommendations else None,
            'alternative_recommendations': recommendations[1:5] if len(recommendations) > 1 else [],
            'all_feasible_options': len(recommendations)
        }

class CarbonOffsetMLModel:
    """
    Machine Learning model for intelligent tree recommendations
    Learns from historical data to provide better recommendations
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.tree_db = TreeSpeciesDatabase()
        self.calculator = CarbonOffsetCalculator()
        
        # ML Models
        self.species_model = None  # Classifier for tree species
        self.count_model = None    # Regressor for tree count
        self.scaler = StandardScaler()
        self.species_encoder = LabelEncoder()
        self.industry_encoder = LabelEncoder()
        
        # Feature names (for reference)
        self.feature_names = [
            'annual_emissions_kg',
            'duration_years',
            'budget_usd',
            'space_m2',
            'latitude',
            'longitude',
            'industry_encoded'
        ]
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def generate_training_data(
        self,
        n_samples: int = 5000,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic training data for ML model
        
        Args:
            n_samples: Number of training samples to generate
            save_path: Optional path to save training data
        
        Returns:
            DataFrame with training data
        """
        
        print(f"Generating {n_samples} training samples...")
        
        # Industry types with typical characteristics
        industries = {
            'manufacturing': {
                'emissions_range': (10000, 1000000),
                'climate_preference': 'temperate',
                'budget_range': (5000, 200000),
                'space_range': (1000, 100000)
            },
            'power_plant': {
                'emissions_range': (500000, 5000000),
                'climate_preference': None,
                'budget_range': (10000, 500000),
                'space_range': (5000, 500000)
            },
            'agriculture': {
                'emissions_range': (5000, 200000),
                'climate_preference': 'tropical',
                'budget_range': (2000, 50000),
                'space_range': (10000, 200000)
            },
            'transportation': {
                'emissions_range': (20000, 500000),
                'climate_preference': 'temperate',
                'budget_range': (10000, 100000),
                'space_range': (5000, 50000)
            },
            'tech_company': {
                'emissions_range': (1000, 50000),
                'climate_preference': 'temperate',
                'budget_range': (5000, 100000),
                'space_range': (1000, 20000)
            }
        }
        
        training_samples = []
        
        for _ in range(n_samples):
            # Random industry selection
            industry_type = np.random.choice(list(industries.keys()))
            industry_info = industries[industry_type]
            
            # Generate random parameters
            annual_emissions = np.random.uniform(*industry_info['emissions_range'])
            duration = np.random.randint(5, 30)
            budget = np.random.uniform(*industry_info['budget_range'])
            space = np.random.uniform(*industry_info['space_range'])
            latitude = np.random.uniform(-60, 60)
            longitude = np.random.uniform(-180, 180)
            climate_pref = industry_info['climate_preference']
            
            # Get optimal recommendation
            result = self.calculator.calculate_optimal_trees(
                annual_emissions_kg=annual_emissions,
                duration_years=duration,
                climate_zone=climate_pref,
                budget_usd=budget,
                space_m2=space,
                optimization_criteria='min_cost'
            )
            
            if result['top_recommendation']:
                recommendation = result['top_recommendation']
                
                training_samples.append({
                    'industry': industry_type,
                    'annual_emissions_kg': annual_emissions,
                    'duration_years': duration,
                    'budget_usd': budget,
                    'space_m2': space,
                    'latitude': latitude,
                    'longitude': longitude,
                    'climate_preference': climate_pref,
                    'recommended_species': recommendation['species_id'],
                    'trees_needed': recommendation['trees_needed'],
                    'total_cost_usd': recommendation['total_cost_usd'],
                    'land_required_m2': recommendation['land_required_m2'],
                    'carbon_offset_kg': recommendation['total_carbon_offset_kg']
                })
        
        df = pd.DataFrame(training_samples)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Training data saved to {save_path}")
        
        print(f"Generated {len(df)} training samples")
        return df
    
    def train(
        self,
        training_data: Optional[pd.DataFrame] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[float, float, float]:
        """
        Train the ML models
        
        Args:
            training_data: DataFrame with training data (generated if None)
            test_size: Proportion for test split
            random_state: Random seed
        
        Returns:
            Tuple of (species_accuracy, count_r2, count_rmse)
        """
        
        if training_data is None:
            training_data = self.generate_training_data(2000)
        
        print(f"\nTraining ML models with {len(training_data)} samples...")
        
        # Prepare features
        X = training_data[[
            'annual_emissions_kg',
            'duration_years',
            'budget_usd',
            'space_m2',
            'latitude',
            'longitude'
        ]].copy()
        
        # Encode industry type
        X['industry_encoded'] = self.industry_encoder.fit_transform(
            training_data['industry']
        )
        
        # Prepare targets
        y_species = training_data['recommended_species']
        y_count = training_data['trees_needed']
        
        # Encode species labels
        y_species_encoded = self.species_encoder.fit_transform(y_species)
        
        # Split data
        X_train, X_test, y_species_train, y_species_test, y_count_train, y_count_test = \
            train_test_split(X, y_species_encoded, y_count, 
                           test_size=test_size, random_state=random_state)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train species classifier (Random Forest)
        print("Training species classifier...")
        self.species_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        self.species_model.fit(X_train_scaled, y_species_train)
        
        # Train count regressor (Gradient Boosting)
        print("Training tree count regressor...")
        self.count_model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=10,
            min_samples_split=5,
            random_state=random_state,
            verbose=0
        )
        self.count_model.fit(X_train_scaled, y_count_train)
        
        # Evaluate models
        species_accuracy = self._evaluate_species_model(X_test_scaled, y_species_test)
        count_r2, count_rmse = self._evaluate_count_model(X_test_scaled, y_count_test)
        
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETE")
        print("="*60)
        print(f"Species Classification Accuracy: {species_accuracy:.2%}")
        print(f"Tree Count Prediction R² Score: {count_r2:.4f}")
        print(f"Tree Count Prediction RMSE: {count_rmse:.2f} trees")
        print("="*60)
        
        return species_accuracy, count_r2, count_rmse
    
    def _evaluate_species_model(self, X_test, y_test) -> float:
        """Evaluate species classification model"""
        y_pred = self.species_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    
    def _evaluate_count_model(self, X_test, y_test) -> Tuple[float, float]:
        """Evaluate tree count regression model"""
        y_pred = self.count_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return r2, rmse
    
    def predict(
        self,
        annual_emissions_kg: float,
        duration_years: int = 10,
        budget_usd: Optional[float] = None,
        space_m2: Optional[float] = None,
        latitude: float = 28.6139,  # Default: Delhi
        longitude: float = 77.2090,
        industry: str = 'manufacturing',
        climate_zone: Optional[str] = None,
        use_ml: bool = True
    ) -> Dict:
        """
        Make prediction for carbon offset requirements
        
        Args:
            annual_emissions_kg: Annual CO2 emissions in kg
            duration_years: Project duration in years
            budget_usd: Optional budget constraint
            space_m2: Optional space constraint
            latitude: Geographical latitude
            longitude: Geographical longitude
            industry: Industry type
            climate_zone: Climate zone for tree suitability
            use_ml: Whether to use ML model or rule-based calculator
        
        Returns:
            Dictionary with prediction results
        """
        
        if use_ml and self.species_model and self.count_model:
            # Use ML model prediction
            return self._predict_ml(
                annual_emissions_kg, duration_years, budget_usd, space_m2,
                latitude, longitude, industry, climate_zone
            )
        else:
            # Use rule-based calculator
            return self._predict_rule_based(
                annual_emissions_kg, duration_years, budget_usd, space_m2,
                climate_zone
            )
    
    def _predict_ml(
        self,
        annual_emissions_kg: float,
        duration_years: int,
        budget_usd: Optional[float],
        space_m2: Optional[float],
        latitude: float,
        longitude: float,
        industry: str,
        climate_zone: Optional[str]
    ) -> Dict:
        """Make prediction using ML model"""
        
        # Prepare input features
        input_data = {
            'annual_emissions_kg': annual_emissions_kg,
            'duration_years': duration_years,
            'budget_usd': budget_usd if budget_usd is not None else 50000,
            'space_m2': space_m2 if space_m2 is not None else 10000,
            'latitude': latitude,
            'longitude': longitude,
            'industry': industry
        }
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        
        # Encode industry
        try:
            input_df['industry_encoded'] = self.industry_encoder.transform([industry])
        except ValueError:
            # If industry not in training, use default
            input_df['industry_encoded'] = 0
        
        # Ensure all feature columns exist
        for col in self.feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Scale features
        input_scaled = self.scaler.transform(input_df[self.feature_names])
        
        # Predict species
        species_encoded = self.species_model.predict(input_scaled)[0]
        species_id = self.species_encoder.inverse_transform([species_encoded])[0]
        
        # Predict tree count
        tree_count = int(self.count_model.predict(input_scaled)[0])
        
        # Get species details
        species_data = self.tree_db.get_species_details(species_id)
        
        if not species_data:
            # Fallback to rule-based if species not found
            return self._predict_rule_based(
                annual_emissions_kg, duration_years, budget_usd, space_m2, climate_zone
            )
        
        # Calculate detailed results
        annual_carbon = species_data['co2_kg_per_year']
        total_carbon_offset = tree_count * annual_carbon * duration_years * 0.75
        
        # Prepare response
        response = {
            'prediction_method': 'ml_model',
            'timestamp': pd.Timestamp.now().isoformat(),
            'input_parameters': {
                'annual_emissions_kg': annual_emissions_kg,
                'duration_years': duration_years,
                'budget_usd': budget_usd,
                'space_m2': space_m2,
                'latitude': latitude,
                'longitude': longitude,
                'industry': industry,
                'climate_zone': climate_zone
            },
            'recommendation': {
                'species_id': species_id,
                'common_name': species_data['common_name'],
                'scientific_name': species_data['scientific_name'],
                'trees_needed': tree_count,
                'annual_carbon_per_tree_kg': annual_carbon,
                'total_carbon_offset_kg': float(total_carbon_offset),
                'project_duration_years': duration_years,
                'total_cost_usd': float(tree_count * species_data['cost_usd']),
                'land_required_m2': float(tree_count * species_data['space_m2']),
                'growth_rate': species_data['growth_rate'],
                'climate_suitability': species_data['climate_zones'],
                'biodiversity_score': species_data['biodiversity_score']
            },
            'carbon_metrics': {
                'input_emissions_kg': annual_emissions_kg * duration_years,
                'offset_achieved_kg': float(total_carbon_offset),
                'offset_percentage': float((total_carbon_offset / (annual_emissions_kg * duration_years)) * 100)
                if annual_emissions_kg * duration_years > 0 else 0,
                'cost_per_kg_co2': float((tree_count * species_data['cost_usd']) / total_carbon_offset)
                if total_carbon_offset > 0 else float('inf')
            },
            'ml_model_info': {
                'species_confidence': float(np.max(self.species_model.predict_proba(input_scaled)[0])),
                'model_type': 'RandomForest + GradientBoosting',
                'training_date': '2024-01-01'  # Would be dynamic in production
            }
        }
        
        return response
    
    def _predict_rule_based(
        self,
        annual_emissions_kg: float,
        duration_years: int,
        budget_usd: Optional[float],
        space_m2: Optional[float],
        climate_zone: Optional[str]
    ) -> Dict:
        """Make prediction using rule-based calculator"""
        
        result = self.calculator.calculate_optimal_trees(
            annual_emissions_kg=annual_emissions_kg,
            duration_years=duration_years,
            climate_zone=climate_zone,
            budget_usd=budget_usd,
            space_m2=space_m2,
            optimization_criteria='min_cost'
        )
        
        response = {
            'prediction_method': 'rule_based',
            'timestamp': pd.Timestamp.now().isoformat(),
            'input_parameters': {
                'annual_emissions_kg': annual_emissions_kg,
                'duration_years': duration_years,
                'budget_usd': budget_usd,
                'space_m2': space_m2,
                'climate_zone': climate_zone
            }
        }
        
        if result['top_recommendation']:
            recommendation = result['top_recommendation']
            response['recommendation'] = recommendation
            response['optimization_criteria'] = result['optimization_criteria']
            response['alternative_options'] = result['alternative_recommendations']
        else:
            response['error'] = 'No feasible solution found within constraints'
        
        return response
    
    def batch_predict(
        self,
        predictions_data: List[Dict],
        use_ml: bool = True
    ) -> List[Dict]:
        """
        Make batch predictions
        
        Args:
            predictions_data: List of dictionaries with prediction parameters
            use_ml: Whether to use ML model
        
        Returns:
            List of prediction results
        """
        
        results = []
        
        for data in predictions_data:
            try:
                result = self.predict(
                    annual_emissions_kg=data.get('annual_emissions_kg', 100000),
                    duration_years=data.get('duration_years', 10),
                    budget_usd=data.get('budget_usd'),
                    space_m2=data.get('space_m2'),
                    latitude=data.get('latitude', 28.6139),
                    longitude=data.get('longitude', 77.2090),
                    industry=data.get('industry', 'manufacturing'),
                    climate_zone=data.get('climate_zone'),
                    use_ml=use_ml
                )
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'input_data': data
                })
        
        return results
    
    def save_model(self, directory: str = 'model_weights'):
        """
        Save trained models to disk
        
        Args:
            directory: Directory to save model files
        """
        
        import os
        os.makedirs(directory, exist_ok=True)
        
        model_data = {
            'species_model': self.species_model,
            'count_model': self.count_model,
            'scaler': self.scaler,
            'species_encoder': self.species_encoder,
            'industry_encoder': self.industry_encoder,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, f'{directory}/carbon_model.pkl')
        print(f"Model saved to {directory}/carbon_model.pkl")
    
    def load_model(self, filepath: str):
        """
        Load trained models from disk
        
        Args:
            filepath: Path to saved model file
        """
        
        model_data = joblib.load(filepath)
        
        self.species_model = model_data['species_model']
        self.count_model = model_data['count_model']
        self.scaler = model_data['scaler']
        self.species_encoder = model_data['species_encoder']
        self.industry_encoder = model_data['industry_encoder']
        self.feature_names = model_data.get('feature_names', self.feature_names)
        
        # print(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict:
        """Get summary of the trained model"""
        
        if not self.species_model or not self.count_model:
            return {'status': 'Model not trained'}
        
        return {
            'status': 'trained',
            'species_model': {
                'type': type(self.species_model).__name__,
                'n_estimators': self.species_model.n_estimators if hasattr(self.species_model, 'n_estimators') else 'N/A',
                'n_features': self.species_model.n_features_in_ if hasattr(self.species_model, 'n_features_in_') else 'N/A'
            },
            'count_model': {
                'type': type(self.count_model).__name__,
                'n_estimators': self.count_model.n_estimators if hasattr(self.count_model, 'n_estimators') else 'N/A'
            },
            'tree_species_count': len(self.tree_db.TREE_DATA),
            'feature_names': self.feature_names
        }
    
    def export_for_unity(self, prediction_result: Dict) -> Dict:
        """
        Format prediction result for Unity integration
        
        Args:
            prediction_result: Prediction result from predict() method
        
        Returns:
            Dictionary formatted for Unity
        """
        
        if 'recommendation' not in prediction_result:
            return {'error': 'No recommendation available'}
        
        rec = prediction_result['recommendation']
        
        # Simple mapping for Unity (can be customized)
        unity_format = {
            'species': {
                'id': rec.get('species_id', ''),
                'common_name': rec.get('common_name', ''),
                'code': rec.get('species_id', '').replace('_', '').lower()[:10]
            },
            'trees': {
                'count': rec.get('trees_needed', 0),
                'spacing_meters': 10.0,  # Can be calculated from land_required
                'layout': 'grid'  # grid, random, organic
            },
            'carbon': {
                'per_tree_kg_per_year': rec.get('annual_carbon_per_tree_kg', 0),
                'total_offset_kg': rec.get('total_carbon_offset_kg', 0),
                'project_years': rec.get('project_duration_years', 10)
            },
            'visualization': {
                'tree_prefab_name': rec.get('species_id', 'oak').lower(),
                'growth_speed': {'slow': 0.3, 'medium': 0.5, 'fast': 0.7, 'very_fast': 0.9}.get(
                    rec.get('growth_rate', 'medium'), 0.5
                ),
                'max_size': rec.get('height_m', 20) / 50.0,  # Normalized for Unity
                'color_variation': 0.2
            },
            'metadata': {
                'prediction_method': prediction_result.get('prediction_method', 'unknown'),
                'timestamp': prediction_result.get('timestamp', ''),
                'constraints_met': 'budget' in prediction_result['input_parameters'] and 
                                  'space' in prediction_result['input_parameters']
            }
        }
        
        return unity_format


# Utility functions for easy usage
def create_and_train_model(
    n_training_samples: int = 3000,
    save_model: bool = True,
    model_dir: str = 'model_weights'
) -> CarbonOffsetMLModel:
    """
    Convenience function to create and train model
    
    Args:
        n_training_samples: Number of training samples
        save_model: Whether to save the trained model
        model_dir: Directory to save model
    
    Returns:
        Trained CarbonOffsetMLModel instance
    """
    
    print("Creating Carbon Offset ML Model...")
    model = CarbonOffsetMLModel()
    
    print(f"Training with {n_training_samples} samples...")
    accuracy, r2, rmse = model.train(
        training_data=None,
        test_size=0.2
    )
    
    print(f"\nTraining Results:")
    print(f"  Species Accuracy: {accuracy:.2%}")
    print(f"  Count R²: {r2:.4f}")
    print(f"  Count RMSE: {rmse:.2f} trees")
    
    if save_model:
        model.save_model(model_dir)
    
    return model

def load_trained_model(model_path: str = 'model_weights/carbon_model.pkl') -> CarbonOffsetMLModel:
    """
    Load a pre-trained model
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Loaded CarbonOffsetMLModel instance
    """
    
    model = CarbonOffsetMLModel(model_path)
    # print(f"Model loaded successfully from {model_path}")
    return model

# Example usage
if __name__ == "__main__":
    # Example 1: Quick training and prediction
    print("="*60)
    print("CARBON OFFSET ML MODEL - EXAMPLE USAGE")
    print("="*60)
    
    # Create and train model
    model = create_and_train_model(n_training_samples=1000, save_model=True)
    
    # Make a prediction
    print("\n" + "="*60)
    print("MAKING PREDICTION")
    print("="*60)
    
    prediction = model.predict(
        annual_emissions_kg=500000,  # 500 tons CO2 per year
        duration_years=15,
        budget_usd=50000,
        space_m2=20000,
        industry='manufacturing',
        climate_zone='temperate',
        use_ml=True
    )
    
    # Print results
    if 'recommendation' in prediction:
        rec = prediction['recommendation']
        print(f"\nRecommended Species: {rec['common_name']} ({rec['scientific_name']})")
        print(f"Number of Trees Needed: {rec['trees_needed']:,}")
        print(f"Total Carbon Offset: {rec['total_carbon_offset_kg']/1000:,.1f} tons CO₂")
        print(f"Total Cost: ${rec['total_cost_usd']:,.2f}")
        print(f"Land Required: {rec['land_required_m2']:,.0f} m²")
        print(f"Growth Rate: {rec['growth_rate'].title()}")
        print(f"Prediction Method: {prediction['prediction_method'].upper()}")
    else:
        print("No recommendation found")
    
    # Export for Unity (when ready)
    print("\n" + "="*60)
    print("UNITY EXPORT FORMAT")
    print("="*60)
    unity_export = model.export_for_unity(prediction)
    print(json.dumps(unity_export, indent=2))