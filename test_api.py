import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_api():
    print("🩺 Testing Medical Disease API\n")
    
    # 1. Health Check
    print("1️⃣ Health Check:")
    try:
        resp = requests.get(f"{BASE_URL}/health")
        print("✅", json.dumps(resp.json(), indent=2))
    except:
        print("❌ API not running!")
        return
    
    # 2. Test Cases
    test_cases = {
        "anemia": [0, 45, 6.5, 1.2, 7500, 22, 25, 210, 3.9, 0.9, 85, 95, 0.01],
        "heart_attack": [1, 65, 12.5, 4.2, 8500, 25, 30, 220, 3.8, 1.1, 95, 100, 0.08],
        "diabetes": [1, 55, 13.2, 4.5, 7800, 28, 35, 215, 4.0, 1.0, 150, 105, 0.02],
        "kidney": [0, 60, 11.8, 4.0, 9200, 30, 28, 218, 3.7, 2.1, 88, 98, 0.03]
    }
    
    print("\n2️⃣ Testing Patients:")
    for name, features in test_cases.items():
        print(f"\n🏥 Patient: {name}")
        resp = requests.post(f"{BASE_URL}/predict", json={"features": features})
        result = resp.json()
        print(json.dumps(result, indent=2))
    
    print("\n✅ All tests completed! 🎉")

if __name__ == "__main__":
    time.sleep(2)  # انتظر API
    test_api()
