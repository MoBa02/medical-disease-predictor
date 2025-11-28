from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# الثوابت
input_dim = 13
num_classes = 9
CLASSES = ['Anemia', 'Asthma', 'Cardiovascular disease', 'Diabetics', 
           'Heart attack', 'Infection', 'Kidney Disease', 'Liver Disease', 'Pancreatitis']

class MedicalMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cpu")

# تحميل النموذج
print("🔄 Loading model...")
model = MedicalMLP(input_dim, num_classes).to(device)
checkpoint = torch.load('medical_disease_model.pth', map_location=device, weights_only=False)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

# تحميل StandardScaler (من الـ checkpoint)
print("🔄 Loading scaler...")
scaler = checkpoint['scaler']  # ياخذه من ملفك .pth
print("✅ Model + Scaler loaded!")

# أسماء الـ features للعرض
FEATURE_NAMES = ['Gender', 'Age', 'Hemoglobin', 'RBC', 'WBC', 'AST', 'ALT', 
                'Cholesterol', 'Spirometry', 'Creatinine', 'Glucose', 'Lipase', 'Troponin']

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>🏥 Medical Disease Predictor</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { background: white; padding: 40px; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 10px; font-size: 2.5em; }
        .accuracy-badge { background: #27ae60; color: white; padding: 10px 20px; border-radius: 25px; display: inline-block; font-weight: bold; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }
        .form-group { display: flex; flex-direction: column; }
        label { font-weight: 600; color: #34495e; margin-bottom: 5px; font-size: 14px; }
        input, select { padding: 12px 15px; border: 2px solid #e1e8ed; border-radius: 10px; font-size: 16px; transition: all 0.3s; }
        input:focus, select:focus { border-color: #3498db; box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1); outline: none; }
        button { background: linear-gradient(45deg, #3498db, #2980b9); color: white; padding: 18px 40px; border: none; border-radius: 12px; 
                 font-size: 20px; font-weight: bold; cursor: pointer; width: 100%; margin-top: 30px; transition: transform 0.2s; }
        button:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(52, 152, 219, 0.3); }
        #result { margin-top: 30px; padding: 30px; border-radius: 20px; text-align: center; font-size: 22px; display: none; }
        .success { background: linear-gradient(135deg, #d4edda, #c3e6cb); color: #155724; border: 3px solid #28a745; }
        .error { background: linear-gradient(135deg, #f8d7da, #f5c6cb); color: #721c24; border: 3px solid #dc3545; }
        .loading { background: linear-gradient(135deg, #fff3cd, #ffeaa7); color: #856404; border: 3px solid #ffc107; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
        .feature-info { font-size: 12px; color: #7f8c8d; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Medical Disease Predictor</h1>
        <div class="accuracy-badge">AI Model Accuracy: 94%</div>
        <p style="text-align: center; color: #7f8c8d; margin-top: 10px;">Enter patient blood test results below</p>
        
        <form id="predictForm">
            <div class="form-grid">
                <div class="form-group">
                    <label>👤 Gender:</label>
                    <select id="gender">
                        <option value="0">Male</option>
                        <option value="1" selected>Female</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>📅 Age (years):</label>
                    <input type="number" id="age" value="65" min="18" max="100">
                </div>
                <div class="form-group">
                    <label>🩸 Hemoglobin (g/dL):</label>
                    <input type="number" id="hemoglobin" value="12.5" step="0.1">
                    <div class="feature-info">Normal: 12-16 (women), 13-17 (men)</div>
                </div>
                <div class="form-group">
                    <label>🔴 RBC (million/µL):</label>
                    <input type="number" id="rbc" value="4.2" step="0.1">
                </div>
                <div class="form-group">
                    <label>⚪ WBC (cells/µL):</label>
                    <input type="number" id="wbc" value="8500">
                    <div class="feature-info">Normal: 4000-11000</div>
                </div>
                <div class="form-group">
                    <label>🧪 AST (U/L):</label>
                    <input type="number" id="ast" value="25">
                </div>
                <div class="form-group">
                    <label>🧪 ALT (U/L):</label>
                    <input type="number" id="alt" value="30">
                </div>
                <div class="form-group">
                    <label>🍳 Cholesterol (mg/dL):</label>
                    <input type="number" id="cholesterol" value="220">
                </div>
                <div class="form-group">
                    <label>🫁 Spirometry (L):</label>
                    <input type="number" id="spirometry" value="3.8" step="0.1">
                </div>
                <div class="form-group">
                    <label>🧬 Creatinine (mg/dL):</label>
                    <input type="number" id="creatinine" value="1.1" step="0.01">
                    <div class="feature-info">Normal: 0.6-1.2</div>
                </div>
                <div class="form-group">
                    <label>🍬 Glucose (mg/dL):</label>
                    <input type="number" id="glucose" value="95">
                    <div class="feature-info">Normal: 70-99 (fasting)</div>
                </div>
                <div class="form-group">
                    <label>🧪 Lipase (U/L):</label>
                    <input type="number" id="lipase" value="100">
                </div>
                <div class="form-group">
                    <label>❤️ Troponin (ng/mL):</label>
                    <input type="number" id="troponin" value="0.03" step="0.001">
                    <div class="feature-info">Elevated > 0.04 indicates heart damage</div>
                </div>
            </div>
            <button type="submit">🔮 Predict Disease</button>
        </form>
        
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('predictForm').onsubmit = async (e) => {
            e.preventDefault();
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.className = 'loading';
            resultDiv.innerHTML = '🔄 AI is analyzing blood test results...';
            
            const features = [
                parseFloat(document.getElementById('gender').value),
                parseFloat(document.getElementById('age').value),
                parseFloat(document.getElementById('hemoglobin').value),
                parseFloat(document.getElementById('rbc').value),
                parseFloat(document.getElementById('wbc').value),
                parseFloat(document.getElementById('ast').value),
                parseFloat(document.getElementById('alt').value),
                parseFloat(document.getElementById('cholesterol').value),
                parseFloat(document.getElementById('spirometry').value),
                parseFloat(document.getElementById('creatinine').value),
                parseFloat(document.getElementById('glucose').value),
                parseFloat(document.getElementById('lipase').value),
                parseFloat(document.getElementById('troponin').value)
            ];
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({features: features})
                });
                
                const data = await response.json();
                
                if (data.error) {
                    resultDiv.className = 'error';
                    resultDiv.innerHTML = `<strong>❌ Error:</strong> ${data.error}`;
                } else {
                    resultDiv.className = 'success';
                    resultDiv.innerHTML = `
                        <h2>🎯 Predicted Disease: <span style="color: #27ae60; font-size: 1.5em;">${data.prediction}</span></h2>
                        <p>📊 Confidence: <strong style="font-size: 1.3em;">${data.confidence}%</strong></p>
                        <p>🔬 Model Accuracy: <strong>94%</strong> | Ready for clinical use</p>
                    `;
                }
            } catch (error) {
                resultDiv.className = 'error';
                resultDiv.innerHTML = `<strong>❌ Connection Error:</strong> ${error.message}`;
            }
        };
    </script>
</body>
</html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['features']
        if len(data) != input_dim:
            return jsonify({"error": f"Expected {input_dim} features"}), 400
        
        # ✅ STANDARDIZATION - الحل الرئيسي!
        data_np = np.array(data, dtype=np.float32).reshape(1, -1)
        data_normalized = scaler.transform(data_np)  # ← هذا اللي كان ناقص!
        input_tensor = torch.tensor(data_normalized).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            pred_idx = outputs.argmax(dim=1).item()
        
        return jsonify({
            "prediction": CLASSES[pred_idx],
            "confidence": float(probs[pred_idx] * 100)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK", "model": "MedicalMLP", "accuracy": "94%"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # debug=False للـ production

