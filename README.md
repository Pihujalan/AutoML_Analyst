# AutoML Analyst 🤖📊

**An intelligent AutoML platform that transforms raw data into actionable business insights through automated machine learning, advanced pattern discovery, and AI-powered analysis.**

## What Problem Does This Solve?

Manual data analysis and model building is time-consuming and requires deep expertise. AutoML Analyst democratizes machine learning by:

- **Eliminating Manual Work**: Automatically cleans data, detects patterns, and builds optimized models
- **Accelerating Time-to-Insight**: Go from raw CSV to business recommendations in minutes, not days
- **Democratizing ML**: No coding required - upload data through a web interface and get expert-level analysis
- **Generating Actionable Intelligence**: AI-powered recommendations in plain English, not just metrics

Perfect for data analysts, business teams, and anyone who needs quick, reliable ML insights without the complexity.

---

## Key Features

### Automated Data Processing
- Intelligent data cleaning and missing value handling
- Smart categorical encoding with relationship preservation
- Automatic feature engineering and selection

### Advanced Pattern Discovery
- Statistical correlation analysis (Pearson, Spearman, Chi-square)
- Multi-method outlier detection (IQR, Z-score, Isolation Forest)
- Time series trend and seasonality detection
- Customer segmentation and cohort analysis

### AutoML Pipeline
- Multiple algorithms: Gradient Boosting, Random Forest, FLAML
- Automated hyperparameter tuning and cross-validation
- Ensemble methods for superior accuracy
- Model performance visualization and comparison

### AI-Powered Business Intelligence
- Natural language insights and explanations
- Actionable recommendations for decision-makers
- Interactive visualizations and reports
- Executive-ready summaries

### Interactive Web Interface
- Drag-and-drop CSV upload or Google Sheets integration
- Visual configuration of data types and targets
- Real-time model training progress
- Downloadable predictions and reports

---

## Features in Detail

### Data Processing
- **Intelligent Cleaning**: Automatic handling of missing values, outliers, and data quality issues
- **Smart Encoding**: Categorical variable encoding with preservation of relationships
- **Feature Engineering**: Automated feature creation and selection
- **Data Validation**: Type detection and consistency checks

### Pattern Discovery
- **Statistical Analysis**: Correlation analysis (Pearson, Spearman), ANOVA, Chi-square tests
- **Outlier Detection**: Multiple methods including IQR, Z-score, and Isolation Forest
- **Time Series Analysis**: Trend detection and seasonal pattern identification
- **Segment Analysis**: Customer/group segmentation and cohort analysis
- **Distribution Analysis**: Normality tests and distribution fitting

### AutoML Pipeline
- **Multiple Algorithms**: Gradient Boosting Machines (GBM), Random Forest, FLAML with automatic selection
- **Hyperparameter Tuning**: Automated optimization of model parameters using grid search and Bayesian optimization
- **Cross-Validation**: Robust k-fold cross-validation for reliable performance estimates
- **Ensemble Methods**: Combining multiple models for superior accuracy
- **Model Comparison**: Automatic selection of best-performing model

### Business Intelligence
- **AI-Powered Insights**: Natural language explanations of findings using advanced language models
- **Recommendation Engine**: Actionable business recommendations based on data patterns
- **Visual Reports**: Interactive charts, graphs, and dashboards
- **Executive Summaries**: High-level insights formatted for decision-makers
- **Contextual Analysis**: Industry-specific insights and benchmarking

### Model Inference & Predictions
- **Interactive Predictions**: CLI-based prediction system for real-time inference
- **Preprocessing Consistency**: Applies the same transformations used during training
- **Encoder Management**: Automatic loading of trained encoders for categorical variables
- **Batch Predictions**: Support for predicting on multiple records at once
- **Confidence Scores**: Probability estimates for classification tasks

---

## Project Structure

```
AutoML_Analyst/
├── data_processing/                 # Core data cleaning & preprocessing modules
│   ├── dataCleaning.py              # Data cleaning and validation logic
│   ├── encodeData.py                # Feature encoding utilities
│   ├── load_dataset.py              # Dataset loading helpers
│   ├── missingPatternAnalysis.py    # Missing-value pattern analysis
│   ├── train_test_split.py          # Train-test splitting logic
│   └── __init__.py
│
├── ml/                              # Machine learning & AutoML pipeline
│   ├── fullmodel.py                 # End-to-end AutoML training pipeline
│   ├── rf.py                        # Random Forest model implementation
│   ├── rfauto.py                    # Automated Random Forest tuning
│   └── __pycache__/
│
├── streamlit_app/                   # Streamlit-based web application
│   ├── streamlit_interface.py       # Main Streamlit UI & orchestration
│   ├── data_processing/             # App-level intermediate data artifacts
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── Y_train.csv
│   │   ├── Y_test.csv
│   │   └── target_config.json
│   │
│   ├── outputs/                     # Generated outputs from training & analysis
│   │   ├── data/                    # Cleaned & encoded datasets
│   │   │   ├── input.csv
│   │   │   ├── cleaned_input.csv
│   │   │   └── encoded_input.csv
│   │   │
│   │   ├── models/                  # Trained ML models
│   │   │   ├── best_rf_model.pkl
│   │   │   ├── final_model.pkl
│   │   │   └── trained_gbm_model_tuned.pkl
│   │   │
│   │   ├── encoders/                # Saved encoders for inference
│   │   │   ├── region_encoder.pkl
│   │   │   ├── sex_encoder.pkl
│   │   │   └── smoker_encoder.pkl
│   │   │
│   │   ├── metrics/                 # Model metrics & summaries
│   │   │   ├── best_rf_metrics.json
│   │   │   └── final_model_summary.json
│   │   │
│   │   ├── plots/                   # Visualizations & SHAP plots
│   │   │   ├── feature_importance_gbm.png
│   │   │   └── shap_summary_gbm.png
│   │   │
│   │   └── reports/                 # Textual analytical reports
│   │       └── graph_descriptions.txt
│
├── tests/                           # Testing & legacy validation data
│   └── legacy_test_data/
│       └── encoders/
│
├── mixtral_integration.py           # LLM-based business intelligence generation
├── model_training.py                # Standalone training utilities
├── requirements.txt                 # Python dependencies
├── env_example.txt                  # Environment variable template
├── LICENSE
└── README.md                        # Project documentation

```

---

## 5-Minute Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- API key from [Together AI](https://together.ai) (free tier available)

### Installation

**1. Clone and Navigate**
```bash
git clone https://github.com/yourusername/automl-analyst.git
cd automl-analyst
```

**2. Set Up Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure API Keys**
```bash
# Copy the example file
copy env_example.txt .env  # Windows
cp env_example.txt .env    # macOS/Linux

# Edit .env and add your API keys
```

**5. Launch the Application**
```bash
streamlit run streamlit_app/streamlit_interface.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## API Key Setup

### Required Keys

This project uses Together AI for generating business intelligence reports. You'll need the following API keys in your `.env` file:

```properties
TOGETHER_API_KEY=your_key_here              # Primary API key for AI analysis
ANALYSIS_API_KEY=your_key_here              # Pattern analysis reports
TRAINING_API_KEY=your_key_here              # Model training insights
FINAL_REPORT_API_KEY=your_key_here          # Final business reports
FINAL_REPORT_API_KEY2=your_key_here         # Backup key 1
FINAL_REPORT_API_KEY3=your_key_here         # Backup key 2
GROQ_API_KEY=your_key_here                  # Alternative LLM provider
```

### How to Get Together AI API Key

1. Visit [together.ai](https://together.ai) and sign up for a free account
2. Navigate to **API Keys** in your dashboard
3. Click **Create New Key**
4. Copy the key and paste it into your `.env` file
5. Free tier includes generous usage limits for testing

---

## Usage Guide

### Using the Web Interface

**Step 1: Upload Data**
- Click "Browse files" or drag-and-drop your CSV file
- Alternatively, connect to Google Sheets with a shareable link

**Step 2: Configure Columns**
- Review auto-detected data types (numerical, categorical, datetime)
- Manually adjust types if needed
- Select your target variable (what you want to predict)

**Step 3: Run Analysis**
- Click "Start AutoML Pipeline"
- Watch real-time progress as the system:
  - Cleans and preprocesses data
  - Discovers patterns and correlations
  - Trains multiple ML models
  - Generates business insights

**Step 4: Review Results**
- Explore interactive visualizations
- Read AI-generated insights in plain English
- Download predictions as CSV
- Export full analysis report as PDF

### Expected Output

After running the pipeline, you'll get:
- **Model Performance Metrics**: Accuracy, precision, recall, F1-score
- **Feature Importance**: Which variables drive predictions
- **Pattern Analysis**: Correlations, outliers, trends
- **Business Recommendations**: Actionable next steps
- **Prediction File**: Ready-to-use predictions for your data

### Programmatic Usage

```python
from ml.fullmodel import AutoMLPipeline

# Initialize the pipeline
pipeline = AutoMLPipeline()

# Load your dataset
pipeline.load_data("customer_churn.csv")

# Preprocess automatically
pipeline.preprocess_data()

# Train models with auto-tuning
pipeline.train_models()

# Generate AI insights
insights = pipeline.generate_business_intelligence()
print(insights)

# Make predictions on new data
predictions = pipeline.predict("new_customers.csv")
```

---

## System Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for dependencies, plus space for datasets
- **Internet**: Required for API-based AI insights

---

## Troubleshooting

### "ModuleNotFoundError" when running the app
**Solution**: Ensure your virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
```

### "API Key Error" or "Unauthorized"
**Solution**: 
1. Check that your `.env` file exists in the project root
2. Verify all API keys are properly formatted (no extra spaces)
3. Confirm your Together AI account is active and has available credits

### "File Not Found" errors
**Solution**: Make sure you're running commands from the project root directory:
```bash
cd AutoML_Analyst  # Navigate to project root first
streamlit run streamlit_app/streamlit_interface.py
```

### Streamlit app won't open
**Solution**:
1. Check if port 8501 is already in use
2. Try manually opening: `http://localhost:8501`
3. Restart the app with: `streamlit run streamlit_app/streamlit_interface.py --server.port 8502`

### Models taking too long to train
**Solution**: 
- Start with a smaller dataset (< 10,000 rows) for testing
- Reduce the number of models in the configuration
- Use a machine with more RAM/CPU cores

### Need more help?
- Check the [GitHub Issues](https://github.com/yourusername/automl-analyst/issues) for similar problems
- Open a new issue with your error message and system details

---

## Testing

Run the test suite to verify your installation:
```bash
python -m pytest tests/
```

For testing predictions with a trained model:
```bash
python test_predictions.py
```

---

## Advanced Configuration

### Custom Model Parameters
Edit `config.json` in the project root to customize:
- Model algorithms to include/exclude
- Hyperparameter search spaces
- Cross-validation folds
- Timeout limits for training

### Data Processing Options
Configure in `data_processing/config.json`:
- Missing value strategies
- Outlier detection thresholds
- Encoding methods for categorical variables
- Feature engineering rules

---

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes and test thoroughly
4. **Commit** with clear messages: `git commit -m 'Add: New pattern detection algorithm'`
5. **Push** to your fork: `git push origin feature/amazing-feature`
6. **Open** a Pull Request with a detailed description

### Contribution Guidelines
- Follow PEP 8 style guidelines for Python code
- Add tests for new features
- Update documentation as needed
- Keep commits focused and atomic

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for full details.

---

## Acknowledgments

Built with powerful open-source tools:
- **FLAML** - AutoML framework by Microsoft Research
- **Streamlit** - Interactive web application framework
- **Together AI** - Language model API for business intelligence
- **Scikit-learn** - Machine learning library
- **Pandas & NumPy** - Data manipulation and analysis

Special thanks to the open-source community for making this project possible.

---

## Support & Contact

- **Email**: support@automl-analyst.com

---

**Made with love for data enthusiasts who want powerful ML without the complexity**


