############################################################
# Enterprise Credit Risk Decision Platform
# Powered by CatBoost + AMD Instinct + ROCm
############################################################

import pandas as pd
import numpy as np
import gradio as gr
import logging
import time

from catboost import (
    CatBoostClassifier,
    Pool,
    EFeaturesSelectionAlgorithm,
    EShapCalcType
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report
)

############################################################
# Logging Setup
############################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("CreditRiskAI")

############################################################
# CONFIGURATION
############################################################

DATA_PATH = "creditcard.csv"   # Credit card fraud dataset
TARGET_COLUMN = "Class"

# Possible target column names for automatic detection
POSSIBLE_TARGETS = [
    "Class",
    "loan_status", 
    "default", 
    "target", 
    "class", 
    "label",
    "risk",
    "bad_loan"
]

############################################################
# DATA INGESTION
############################################################

def load_dataset():

    logger.info("Loading Kaggle Financial Dataset")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna()

    logger.info(f"Dataset loaded with {df.shape[0]} rows")

    return df

############################################################
# DATA PREPARATION
############################################################

def detect_target_column(df):

    for col in POSSIBLE_TARGETS:
        if col in df.columns:
            logger.info(f"Detected target column: {col}")
            return col

    raise ValueError(
        f"No known target column found. Columns available: {df.columns.tolist()}"
    )


def prepare_data(df):

    target_column = detect_target_column(df)

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Convert German Credit labels
    if set(y.unique()) == {1, 2}:
        y = y.map({1: 0, 2: 1})

    # Convert categorical targets
    if y.dtype == object:
        y = y.astype("category").cat.codes

    categorical_cols = [
        c for c in X.columns if X[c].dtype == "object"
    ]

    return X, y, categorical_cols



############################################################
# FEATURE SELECTION
############################################################

def perform_feature_selection(train_pool, X):

    logger.info("Running Feature Selection")

    selector = CatBoostClassifier(iterations=60, verbose=False)

    try:
        summary = selector.select_features(
            train_pool,
            features_for_select=list(range(X.shape[1])),
            num_features_to_select=min(10, X.shape[1]),
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
            shap_calc_type=EShapCalcType.Regular,
            train_final_model=False
        )
        
        selected_features = [
            X.columns[i] for i in summary["selected_features"]
        ]
    except AttributeError:
        # Fallback to simple feature importance selection
        logger.info("Using feature importance fallback for selection")
        selector.fit(train_pool)
        feature_importance = selector.get_feature_importance()
        top_indices = np.argsort(feature_importance)[-10:]  # Top 10 features
        selected_features = [X.columns[i] for i in top_indices]

    return selected_features

# MODEL TRAINING (GPU vs CPU)
############################################################

def train_credit_model(X, y, categorical_cols, use_gpu=True, iterations=500):

    logger.info("Splitting dataset")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # Changed to 80/20 split
    )

    train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
    test_pool = Pool(X_test, y_test, cat_features=categorical_cols)

    device_type = "GPU" if use_gpu else "CPU"
    logger.info(f"Starting CatBoost {device_type} training")

    start_time = time.time()

    if use_gpu:
        model = CatBoostClassifier(
            iterations=iterations,
            depth=8,                    # Increased depth to utilize GPU better
            learning_rate=0.05,         # Lower LR for more iterations
            loss_function="Logloss",
            eval_metric="Logloss",
            task_type="GPU",
            devices="0",
            gpu_ram_part=0.95,         # Use more GPU memory for better performance
            border_count=254,          # Optimize for GPU
            thread_count=1,           # Use all available threads
            verbose=50
        )
    else:
        model = CatBoostClassifier(
            iterations=iterations,
            depth=8,                    # Match GPU depth
            learning_rate=0.05,         # Match GPU learning rate
            loss_function="Logloss",
            eval_metric="AUC",
            task_type="CPU",
            thread_count=1,         
            verbose=50
        )

    model.fit(train_pool)

    training_time = time.time() - start_time

    logger.info(f"Training completed in {training_time:.2f}s")

    ################################################
    # Evaluation
    ################################################
    preds = model.predict_proba(test_pool)[:, 1]
    auc = roc_auc_score(y_test, preds)

    cm = confusion_matrix(y_test, preds > 0.5)
    report = classification_report(y_test, preds > 0.5)

    ################################################
    # Feature Ranking
    ################################################
    feature_importance = model.get_feature_importance(train_pool)
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False)

    ################################################
    # Recursive Feature Selection
    ################################################
    selected_features = perform_feature_selection(train_pool, X)

    metrics = {
        "AUC": auc,
        "Confusion Matrix": cm,
        "Classification Report": report,
        "Training Time": training_time,
        "Device": device_type,
        "Iterations": iterations
    }

    return model, importance_df, selected_features, metrics

############################################################
# MODEL TESTING FUNCTIONS
############################################################

def evaluate_test_set(model, X_test, y_test, cat_cols):
    """Evaluate model on test set"""
    
    test_pool = Pool(X_test, y_test, cat_features=cat_cols)
    
    # Get predictions
    test_preds = model.predict_proba(test_pool)[:, 1]
    test_binary_preds = (test_preds > 0.5).astype(int)
    
    # Calculate metrics
    test_auc = roc_auc_score(y_test, test_preds)
    test_cm = confusion_matrix(y_test, test_binary_preds)
    test_report = classification_report(y_test, test_binary_preds, output_dict=True)
    
    # Calculate detailed metrics
    accuracy = (test_cm[0][0] + test_cm[1][1]) / test_cm.sum()
    precision = test_cm[1][1] / (test_cm[1][1] + test_cm[0][1]) if (test_cm[1][1] + test_cm[0][1]) > 0 else 0
    recall = test_cm[1][1] / (test_cm[1][1] + test_cm[1][0]) if (test_cm[1][1] + test_cm[1][0]) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    test_results = {
        "Test AUC": f"{test_auc:.4f}",
        "Test Accuracy": f"{accuracy:.4f}",
        "Test Precision": f"{precision:.4f}",
        "Test Recall": f"{recall:.4f}",
        "Test F1-Score": f"{f1_score:.4f}",
        "True Negatives": int(test_cm[0][0]),
        "False Positives": int(test_cm[0][1]),
        "False Negatives": int(test_cm[1][0]),
        "True Positives": int(test_cm[1][1]),
        "Total Test Samples": len(y_test)
    }
    
    return test_results, test_preds

# Store test data globally for the testing tab
X_train_global, X_test_global, y_train_global, y_test_global = None, None, None, None

def store_test_data(X, y, cat_cols):
    """Store test data for evaluation"""
    global X_train_global, X_test_global, y_train_global, y_test_global
    X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

############################################################
# BENCHMARK COMPARISON
############################################################

def run_benchmark_comparison(X, y, cat_cols, iterations=1000):
    """Compare GPU vs CPU performance"""
    
    results = {}
    
    # Use larger subset for better GPU utilization
    if len(X) > 50000:
        sample_size = min(100000, len(X))
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
    else:
        X_sample, y_sample = X, y
        
    logger.info(f"Using {len(X_sample)} samples for benchmark")
    
    # Run CPU benchmark
    logger.info("Running CPU Benchmark...")
    cpu_model, cpu_importance, cpu_features, cpu_metrics = train_credit_model(
        X_sample, y_sample, cat_cols, use_gpu=False, iterations=iterations
    )
    
    results["CPU"] = {
        "Runtime": cpu_metrics["Training Time"],
        "AUC": cpu_metrics["AUC"], 
        "Accuracy": ((cpu_metrics["Confusion Matrix"][0][0] + 
                     cpu_metrics["Confusion Matrix"][1][1]) / 
                     cpu_metrics["Confusion Matrix"].sum()),
        "Precision": (cpu_metrics["Confusion Matrix"][1][1] / 
                     (cpu_metrics["Confusion Matrix"][1][1] + 
                      cpu_metrics["Confusion Matrix"][0][1])),
        "Recall": (cpu_metrics["Confusion Matrix"][1][1] / 
                  (cpu_metrics["Confusion Matrix"][1][1] + 
                   cpu_metrics["Confusion Matrix"][1][0]))
    }
    
    # Run GPU benchmark  
    logger.info("Running GPU Benchmark...")
    gpu_model, gpu_importance, gpu_features, gpu_metrics = train_credit_model(
        X_sample, y_sample, cat_cols, use_gpu=True, iterations=iterations
    )
    
    results["GPU"] = {
        "Runtime": gpu_metrics["Training Time"],
        "AUC": gpu_metrics["AUC"],
        "Accuracy": ((gpu_metrics["Confusion Matrix"][0][0] + 
                     gpu_metrics["Confusion Matrix"][1][1]) / 
                     gpu_metrics["Confusion Matrix"].sum()),
        "Precision": (gpu_metrics["Confusion Matrix"][1][1] / 
                     (gpu_metrics["Confusion Matrix"][1][1] + 
                      gpu_metrics["Confusion Matrix"][0][1])),
        "Recall": (gpu_metrics["Confusion Matrix"][1][1] / 
                  (gpu_metrics["Confusion Matrix"][1][1] + 
                   gpu_metrics["Confusion Matrix"][1][0]))
    }
    
    return results, cpu_importance, gpu_importance

############################################################
# BUSINESS KPI ENGINE
############################################################

def calculate_business_kpis(df):

    fraud_rate = df[TARGET_COLUMN].mean()

    avg_amount = df["Amount"].mean() if "Amount" in df.columns else 0

    portfolio_risk = fraud_rate * 100

    return {
        "Fraud Detection Rate (%)": round(portfolio_risk, 2),
        "Average Transaction Amount": round(avg_amount, 2)
    }

############################################################
# LOAD + TRAIN (Initial GPU Model)
############################################################

dataset = load_dataset()
X, y, cat_cols = prepare_data(dataset)

# Store test data for later evaluation
store_test_data(X, y, cat_cols)

# Train initial GPU model for quick start
model, importance_df, selected_features, metrics = train_credit_model(
    X, y, cat_cols, use_gpu=True, iterations=500
)

business_kpis = calculate_business_kpis(dataset)

############################################################
# PREDICTION FUNCTION
############################################################

def predict_fraud_risk(*inputs, current_model=None):
    
    if current_model is None:
        current_model = model
        
    row = pd.DataFrame([inputs], columns=X.columns)
    prob = current_model.predict_proba(row)[0][1]

    risk_level = "LOW"
    if prob > 0.6:
        risk_level = "HIGH"
    elif prob > 0.3:
        risk_level = "MEDIUM"

    return (
        f"{prob:.2%}",
        risk_level
    )

def retrain_model(use_gpu, iterations_val):
    """Retrain model with selected device"""
    global model, importance_df, selected_features, metrics, X_test_global, y_test_global
    
    logger.info(f"Retraining model on {'GPU' if use_gpu else 'CPU'}")
    
    # Re-split data for new training
    store_test_data(X, y, cat_cols)
    
    model, importance_df, selected_features, metrics = train_credit_model(
        X, y, cat_cols, use_gpu=use_gpu, iterations=int(iterations_val)
    )
    
    return (
        f"Model retrained on {'GPU' if use_gpu else 'CPU'}",
        f"Training Time: {metrics['Training Time']:.2f}s",
        f"AUC Score: {metrics['AUC']:.4f}",
        importance_df
    )

############################################################
# GRADIO ENTERPRISE UI
############################################################

custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global font improvements */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Monospace for code/data */
    code, .gr-textbox textarea, .gr-number input {
        font-family: 'JetBrains Mono', 'SF Mono', Monaco, Consolas, monospace !important;
        font-weight: 500 !important;
    }
    
    /* Professional headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        line-height: 1.2 !important;
    }
    
    /* AMD brand colors and professional styling */
    .gradio-container {
        font-size: 14px !important;
        line-height: 1.6 !important;
        color: #1a1a1a !important;
    }
    
    /* Professional button styling */
    .gr-button-primary {
        background: linear-gradient(135deg, #ED1C24 0%, #C5282F 100%) !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        letter-spacing: 0.02em !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(237, 28, 36, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    
    .gr-button-primary:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(237, 28, 36, 0.4) !important;
    }
    
    /* Professional text inputs */
    .gr-textbox label, .gr-dropdown label, .gr-number label {
        font-weight: 600 !important;
        font-size: 13px !important;
        color: #2c3e50 !important;
        letter-spacing: 0.01em !important;
    }
    
    /* Hide the default dropdown button for tabs */
    .gradio-tabs button[data-tab-id]:not([data-tab-id*="__tab"]) ~ button:last-child {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Force all tab buttons to be visible */
    .gradio-tabs .tab-nav button {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* Ensure all tabs are visible and container uses full width */
    .gradio-tabs .tab-nav {
        height: auto !important;
        max-height: none !important;
        width: 100% !important;
    }
    
    /* Professional card styling */
    .gr-box {
        border-radius: 12px !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08) !important;
        border: 1px solid rgba(0, 0, 0, 0.06) !important;
    }
    
    /* Enhanced markdown styling */
    .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
        font-weight: 700 !important;
        color: #1a1a1a !important;
    }
    
    .gr-markdown p {
        font-size: 14px !important;
        line-height: 1.7 !important;
        color: #4a5568 !important;
    }
    
    .gr-markdown strong {
        font-weight: 600 !important;
        color: #2d3748 !important;
    }
"""

with gr.Blocks(title="Enterprise Credit Card Fraud Detection Platform", theme=gr.themes.Soft(), css=custom_css) as demo:

    # Professional AMD header with enhanced typography
    gr.HTML("""
        <div style="position: relative; padding: 24px 28px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 16px; margin-bottom: 24px; border: 1px solid rgba(0,0,0,0.06);">
            <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg" alt="AMD Logo" style="position: absolute; top: 20px; right: 28px; height: 40px; width: auto;" />
            <div style="padding-right: 140px;">
                <h1 style="margin: 0; color: #1a1a1a; font-size: 2.4em; font-weight: 800; font-family: 'Inter', sans-serif; letter-spacing: -0.03em; line-height: 1.1;">CatBoost Fraud Detection Platform</h1>
                <h3 style="margin: 8px 0 0 0; color: #ED1C24; font-size: 1.1em; font-weight: 600; font-family: 'Inter', sans-serif; letter-spacing: 0.01em;">Powered by CatBoost + AMD Instinct + ROCm</h3>
                <p style="margin: 4px 0 0 0; color: #6b7280; font-size: 0.9em; font-weight: 400;">Enterprise-Grade GPU-Accelerated Credit Card Fraud Detection</p>
            </div>
        </div>
    """)

    gr.Markdown("""
    ### Advanced Features:
    - **CatBoost Explainable AI** - State-of-the-art gradient boosting with feature importance
    - **AMD Instinct™ GPU Acceleration** - High-performance computing for faster training
    - **ROCm™ Open AI Stack** - Optimized for AMD hardware acceleration
    - **Real-time Fraud Detection** - Enterprise-grade performance and accuracy
    - **80/20 Train/Test Split** - Rigorous evaluation on hold-out test data

    ---
    """)

    ################################################
    # Executive Dashboard
    ################################################
    with gr.Tab("Executive KPI Dashboard"):

        gr.Markdown("## Fraud Detection Overview")

        with gr.Row():
            for k, v in business_kpis.items():
                gr.Markdown(f"**{k}:** {v}")

        gr.Markdown("## Current Model Performance")
        
        with gr.Row():
            gr.Markdown(f"**Device:** {metrics['Device']}")
            gr.Markdown(f"**ROC-AUC:** {metrics['AUC']:.4f}")
            gr.Markdown(f"**Training Time:** {metrics['Training Time']:.2f} sec")
            gr.Markdown(f"**Iterations:** {metrics['Iterations']}")

    ################################################
    # GPU vs CPU Benchmark
    ################################################ 
    with gr.Tab("🚀 GPU vs CPU Benchmark"):
        
        gr.Markdown("## CatBoost Performance Comparison")
        gr.Markdown("*Compare AMD Instinct GPU vs CPU performance on fraud detection*")
        
        benchmark_btn = gr.Button("Run Benchmark Comparison", variant="primary")
        benchmark_iterations = gr.Slider(500, 2000, value=1000, label="Training Iterations")
        
        with gr.Row():
            cpu_results = gr.JSON(label="CPU Results")
            gpu_results = gr.JSON(label="GPU Results")
            
        benchmark_status = gr.Textbox(label="Benchmark Status")
        
        def run_benchmark(iterations):
            try:
                results, cpu_imp, gpu_imp = run_benchmark_comparison(X, y, cat_cols, iterations)
                speedup = results['CPU']['Runtime']/results['GPU']['Runtime']
                speedup_text = f"GPU Speedup: {speedup:.2f}x"
                if speedup > 1:
                    speedup_text += " Faster"
                else:
                    speedup_text += " ⚠️ (GPU slower - try more iterations)"
                    
                return (
                    results["CPU"], 
                    results["GPU"], 
                    f"Benchmark Complete! {speedup_text}"
                )
            except Exception as e:
                return {}, {}, f"Error: {str(e)}"
        
        benchmark_btn.click(
            run_benchmark,
            inputs=[benchmark_iterations],
            outputs=[cpu_results, gpu_results, benchmark_status]
        )

    ################################################
    # Model Training Control
    ################################################
    with gr.Tab("⚙️ Model Training"):
        
        gr.Markdown("## Retrain Model")
        
        with gr.Row():
            device_choice = gr.Radio(["CPU", "GPU"], value="GPU", label="Training Device")
            training_iterations = gr.Slider(50, 1000, value=200, label="Training Iterations")
        
        retrain_btn = gr.Button("Retrain Model", variant="primary")
        
        with gr.Column():
            train_status = gr.Textbox(label="Training Status")
            train_time = gr.Textbox(label="Training Metrics") 
            train_auc = gr.Textbox(label="Model Performance")
            updated_importance = gr.DataFrame(label="Updated Feature Importance")
        
        def handle_retrain(device, iterations):
            use_gpu = (device == "GPU")
            return retrain_model(use_gpu, iterations)
        
        retrain_btn.click(
            handle_retrain,
            inputs=[device_choice, training_iterations],
            outputs=[train_status, train_time, train_auc, updated_importance]
        )

    ################################################
    # Feature Explainability
    ################################################
    with gr.Tab("🔍 Explainability"):

        gr.Markdown("## Feature Importance Ranking")
        gr.DataFrame(importance_df)

        gr.Markdown("## Selected Predictive Features")
        gr.Textbox(", ".join(selected_features))

    ################################################
    # Test Set Evaluation
    ################################################
    with gr.Tab("🧪 Test Set Evaluation"):
        
        gr.Markdown("## Model Testing on Hold-out Data (20% of dataset)")
        gr.Markdown("*Evaluate model performance on unseen test data*")
        
        test_btn = gr.Button("Evaluate on Test Set", variant="primary")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Test Set Metrics")
                test_metrics_output = gr.JSON(label="Test Performance Metrics")
                
            with gr.Column():
                gr.Markdown("### Test Set Info")
                test_info = gr.Textbox(label="Test Set Information", 
                                     value=f"Test set size: {len(y_test_global) if y_test_global is not None else 'Not loaded'} samples")
        
        test_status = gr.Textbox(label="Evaluation Status")
        
        def run_test_evaluation():
            global model, X_test_global, y_test_global, cat_cols
            try:
                if X_test_global is None or y_test_global is None:
                    return {}, "Error: Test data not available"
                
                test_results, test_preds = evaluate_test_set(model, X_test_global, y_test_global, cat_cols)
                
                return (
                    test_results, 
                    f"✅ Test evaluation completed successfully! Evaluated {len(y_test_global)} samples."
                )
            except Exception as e:
                return {}, f"❌ Error during test evaluation: {str(e)}"
        
        test_btn.click(
            run_test_evaluation,
            inputs=[],
            outputs=[test_metrics_output, test_status]
        )

    ################################################
    # Risk Prediction Tool
    ################################################
    with gr.Tab("🎯 Transaction Fraud Simulator"):

        inputs = []

        for col in X.columns:
            if col in cat_cols:
                unique_vals = dataset[col].dropna().unique().tolist()
                inputs.append(
                    gr.Dropdown(
                        unique_vals,
                        label=col,
                        value=unique_vals[0] if unique_vals else None
                    )
                )
            else:
                inputs.append(
                    gr.Number(
                        label=col,
                        value=float(dataset[col].median())
                    )
                )

        predict_btn = gr.Button("Predict Fraud Risk", variant="primary")

        prob_output = gr.Textbox(label="Fraud Probability")
        risk_output = gr.Textbox(label="Risk Tier")

        predict_btn.click(
            predict_fraud_risk,
            inputs=inputs,
            outputs=[prob_output, risk_output]
        )

############################################################
# Launch UI
############################################################

demo.launch(server_port=7866, server_name="0.0.0.0")
