
import sys
import threading
import time
import requests
import json
from hi import CarbonOffsetMLModel, CarbonOffsetAPI

def run_server():
    print("Starting Test API Server...")
    # Initialize model (quick training for test)
    model = CarbonOffsetMLModel()
    X, y_species, y_count, _ = model.prepare_training_data(n_samples=100)
    model.train_models(X, y_species, y_count, test_size=0.2)
    
    # Initialize API
    api = CarbonOffsetAPI(model)
    
    # Run server
    # Disable reloader to run in thread
    api.app.run(host='localhost', port=5000, debug=False, use_reloader=False)

def test_prediction():
    # Wait for server to start
    time.sleep(5)
    
    print("\n" + "="*50)
    print("TESTING UNITY CONNECTION LOGIC")
    print("="*50)
    
    url = "http://localhost:5000/predict"
    
    # Payload matching Unity's IndustryData class
    payload = {
        "Industry_Type": "Manufacturing",
        "Annual_Emissions_kg": 50000.0,
        "Duration_Years": 10,
        "Available_Space_m2": 15000.0,
        "Budget_USD": 30000.0,
        "Latitude": 28.6139,
        "Longitude": 77.2090,
        "Climate_Preference": "Tropical",
        "include_optimization": True,
        "Optimization_Criteria": "min_cost"
    }
    
    print(f"Sending request to {url}...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print("\nSUCCESS: Connection Established!")
            data = response.json()
            
            print("\nResponse Received:")
            print(json.dumps(data, indent=2))
            
            # Verify fields expected by Unity
            print("\nVerifying Unity Integration Fields:")
            
            unity_data = data.get('unity_integration')
            if unity_data:
                print(f"✓ species_code: {unity_data.get('species_code')}")
                print(f"✓ tree_count: {unity_data.get('tree_count')}")
                print(f"✓ spacing_m: {unity_data.get('spacing_m')}")
                print(f"✓ growth_rate_unity: {unity_data.get('growth_rate_unity')}")
                print(f"✓ max_height_unity: {unity_data.get('max_height_unity')}")
                print("Unity Integration Data matches C# class structure.")
            else:
                print("❌ Missing 'unity_integration' field!")
                
        else:
            print(f"\n❌ Error: Server returned status code {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        
    print("\nTest Complete.")
    
    # Force exit since Flask runs in a thread
    import os
    os._exit(0)

if __name__ == "__main__":
    # Start server in separate thread
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Run test
    test_prediction()
