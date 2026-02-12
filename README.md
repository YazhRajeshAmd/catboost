# CatBoost Fraud Detection Platform

Enterprise-grade credit card fraud detection platform powered by CatBoost, AMD Instinct GPUs, and ROCm.

## Features

- **CatBoost Explainable AI**: State-of-the-art gradient boosting with feature importance analysis
- **AMD Instinct™ GPU Acceleration**: High-performance computing for faster training
- **ROCm™ Open AI Stack**: Optimized for AMD hardware acceleration  
- **Real-time Fraud Detection**: Enterprise-grade performance and accuracy
- **80/20 Train/Test Split**: Rigorous evaluation on hold-out test data
- **Interactive Web UI**: Professional Gradio interface with AMD branding
- **GPU vs CPU Benchmarking**: Performance comparison tools
- **Model Retraining**: Dynamic device selection and hyperparameter tuning

## Prerequisites

### Hardware Requirements
- AMD Instinct GPU (MI200/MI300 series recommended)
- ROCm-compatible system
- Minimum 8GB RAM
- 16GB+ GPU memory (for large datasets)

### Software Requirements
- Python 3.8+
- ROCm 5.0+
- CUDA-compatible drivers (if using NVIDIA fallback)

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd catboost_demo
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install CatBoost with GPU Support
```bash
pip install catboost-1.2.8-cp312-cp312-linux_x86_64.whl
```

## Dataset

The platform uses the **Credit Card Fraud Detection** dataset:
- **Format**: CSV with 284,807 transactions
- **Features**: 30 anonymized features (V1-V28) + Time, Amount
- **Target**: Binary classification (0=Normal, 1=Fraud)
- **Split**: 80% training, 20% testing

### Dataset Setup
1. Download the `creditcard.csv` dataset
2. Place it in the project root directory
3. The application will automatically load and preprocess the data

## Usage

### Start the Application
```bash
python3 catboost_demo.py
```

The web interface will be available at: `http://localhost:7866`

### Key Functionality

#### 1. **Executive Dashboard** 
- View portfolio fraud detection metrics
- Monitor model performance statistics
- Real-time KPIs and business intelligence

#### 2. **GPU vs CPU Benchmark** 
- Compare training performance across devices
- Adjustable iteration counts (500-2000)
- Detailed speedup analysis

#### 3. **Model Training** 
- Switch between GPU/CPU training
- Customize hyperparameters
- Live performance monitoring

#### 4. **Explainability** 
- Feature importance rankings
- Selected predictive features
- Model interpretability insights

#### 5. **Test Set Evaluation** 
- Hold-out test performance
- Detailed metrics (AUC, Precision, Recall, F1)
- Confusion matrix analysis

#### 6. **Fraud Simulation** 
- Interactive transaction simulator
- Real-time fraud risk prediction
- Risk tier classification (LOW/MEDIUM/HIGH)

## Configuration

### GPU Settings
```python
# GPU Configuration
gpu_ram_part=0.95          # Use 95% of GPU memory
border_count=254           # Optimize for GPU processing
depth=8                    # Model depth for complexity
iterations=500-2000        # Training iterations
```

### Model Parameters
```python
# CatBoost Parameters
learning_rate=0.05         # Learning rate
loss_function="Logloss"    # Loss function
eval_metric="Logloss"      # Evaluation metric
task_type="GPU"           # Device type
```

## Performance Benchmarks

### Expected GPU Speedups
- **MI300X**: 2-5x faster than CPU
- **MI250X**: 1.5-3x faster than CPU
- **Performance depends on**: Dataset size, iterations, model complexity

### Optimization Tips
- Use higher iteration counts for better GPU utilization
- Increase dataset size for maximum GPU benefit
- Adjust `gpu_ram_part` based on available memory

## UI Customization

The interface uses professional AMD branding:
- **Font**: Inter typography family
- **Colors**: AMD red (#ED1C24) primary buttons
- **Layout**: Modern card-based design
- **Responsive**: Works on desktop and mobile

## Troubleshooting

### Common Issues

**GPU not detected:**
```bash
# Check ROCm installation
rocm-smi
# Verify GPU visibility
python -c "import catboost; print(catboost.get_gpu_device_count())"
```

**Memory errors:**
- Reduce `gpu_ram_part` to 0.6-0.8
- Decrease model `depth` parameter
- Use smaller datasets for testing

**Slow performance:**
- Increase `iterations` for better GPU utilization
- Verify ROCm drivers are properly installed
- Check system thermal throttling

## Project Structure

```
catboost_demo/
├── catboost_demo.py           # Main application
├── creditcard.csv             # Dataset
├── requirements.txt           # Dependencies
├── catboost-1.2.8-*.whl      # CatBoost wheel
├── README.md                  # Documentation
└── screenshots/               # UI screenshots
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

## Acknowledgments

- **AMD**: ROCm platform and Instinct GPU technology
- **CatBoost**: Yandex's gradient boosting framework
- **Gradio**: Web interface framework
- **Scikit-learn**: Machine learning utilities
- **Pandas/NumPy**: Data processing libraries

## Contact

For questions, issues, or enterprise support:
- Create GitHub issue
- Contact AMD Developer Support
- ROCm Community Forums

---

**Powered by AMD Instinct + ROCm + CatBoost**
