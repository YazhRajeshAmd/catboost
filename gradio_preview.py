"""
Preview shim — returns realistic dummy data without creditcard.csv or a GPU.
Accepts the same 30-column input as catboost_demo.py so the React frontend
works identically against this as it does against the real backend.

Usage:
    pip install gradio
    python3 gradio_preview.py
    # Then: cd frontend && npm run dev
"""

import gradio as gr

# Column order matches creditcard.csv: Time, V1–V28, Amount
COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

def predict_fraud_risk(*inputs):
    values = dict(zip(COLUMNS, inputs))

    # Simple heuristic using the two strongest real-world signals
    v14     = float(values.get("V14", 0))
    v17     = float(values.get("V17", 0))
    amount  = float(values.get("Amount", 0))
    v1      = float(values.get("V1", 0))

    score = 0.05
    if v14 < -5:   score += 0.40
    if v17 < -10:  score += 0.35
    if amount > 1000: score += 0.15
    if v1 < -2:    score += 0.10

    score = min(max(score, 0.01), 0.99)

    risk_tier = "LOW"
    if score > 0.6:   risk_tier = "HIGH"
    elif score > 0.3: risk_tier = "MEDIUM"

    return f"{score:.2%}", risk_tier


inputs = [gr.Number(label=col, value=0.0) for col in COLUMNS]

demo = gr.Interface(
    fn=predict_fraud_risk,
    inputs=inputs,
    outputs=[
        gr.Textbox(label="Fraud Probability"),
        gr.Textbox(label="Risk Tier"),
    ],
    title="CatBoost Fraud Detection — Preview Mode",
    description="Mock backend. Returns heuristic scores based on V14, V17, and Amount. No dataset or GPU required.",
)

if __name__ == "__main__":
    print("Preview backend running at http://localhost:7866")
    print("Load Suspicious Sample in the UI → expect HIGH risk")
    print("Load Normal Sample in the UI    → expect LOW risk")
    demo.launch(server_port=7866, server_name="0.0.0.0")
