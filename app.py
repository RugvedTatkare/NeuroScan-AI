from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import onnxruntime as rt
import io

app = Flask(__name__)

model1 = rt.InferenceSession("brain_tumor_model.onnx")
model2 = rt.InferenceSession("tumor_type_model.onnx")

classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = image.resize((128, 128))
        img = np.array(img).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # Model 1 - Tumor or Not
        input1 = model1.get_inputs()[0].name
        pred1 = model1.run(None, {input1: img})[0][0][0]

        if pred1 > 0.5:
            result = "Tumor Detected"
            confidence = round(float(pred1) * 100, 2)

            # Model 2 - Tumor Type
            input2 = model2.get_inputs()[0].name
            pred2 = model2.run(None, {input2: img})[0][0]
            tumor_type = classes[np.argmax(pred2)]
            type_confidence = round(float(np.max(pred2)) * 100, 2)
        else:
            result = "No Tumor Detected"
            confidence = round((1 - float(pred1)) * 100, 2)
            tumor_type = None
            type_confidence = None

        return jsonify({
            "result": result,
            "confidence": confidence,
            "tumor_type": tumor_type,
            "type_confidence": type_confidence
        })

    except Exception as e:
        return jsonify({"result": f"Error: {str(e)}", "confidence": 0})

if __name__ == "__main__":
    app.run(debug=False)