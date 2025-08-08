# AutoML Analyst 🤖📊

A comprehensive AutoML solution that combines automated machine learning with advanced data analysis, pattern discovery, and business intelligence generation.

## 🚀 Features

- **Automated Data Processing**: Intelligent data cleaning, encoding, and preprocessing
- **Advanced Pattern Discovery**: Statistical analysis, correlation studies, and outlier detection
- **AutoML Pipeline**: Automated model training with multiple algorithms (GBM, Random Forest, FLAML)
- **Business Intelligence**: AI-powered insights and recommendations
- **Interactive Streamlit Interface**: User-friendly web application
- **Comprehensive Reporting**: Detailed analysis reports and visualizations

## 📁 Project Structure

```
AutoML_Analyst/
├── data_processing/          # Data cleaning and preprocessing modules
│   ├── dataCleaning.py       # Main data cleaning logic
│   ├── encodeData.py         # Data encoding utilities
│   ├── patternDiscovery.py   # Pattern analysis functions
│   └── encoders/             # Encoder storage directory
├── streamlit_app/           # Streamlit web interface
│   └── streamlit_interface.py # Main Streamlit application
├── ml/                      # Machine learning models and training
│   ├── fullmodel.py         # Complete AutoML pipeline
│   ├── rf.py                # Random Forest implementation
│   └── rfauto.py            # Automated Random Forest
├── datasets/                # Data storage directory
├── mixtral_integration.py   # AI-powered business intelligence
├── merged_pattern_analysis.py # Advanced pattern analysis
├── model_training.py        # Model training utilities
├── test_predictions.py      # Prediction testing utilities
├── requirements.txt         # Python dependencies
├── env_example.txt          # Environment variables template
└── README.md               # This file
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/automl-analyst.git
   cd automl-analyst
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env_example.txt .env
   # Edit .env with your API keys
   ```

## 🔑 Required API Keys

You'll need the following API keys (see `env_example.txt` for details):
- Together AI API Key
- Analysis API Key
- Training API Key
- Final Report API Keys

## 🚀 Quick Start

1. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app/streamlit_interface.py
   ```

2. **Upload your dataset** through the web interface

3. **Configure data types** and target variable

4. **Run the AutoML pipeline** to get insights and predictions

## 📊 Usage

### Web Interface
- Upload CSV files or connect to Google Sheets
- Configure column data types
- Select target variable for prediction
- View comprehensive analysis reports
- Download predictions and insights

### Programmatic Usage
```python
from ml.fullmodel import AutoMLPipeline

# Initialize pipeline
pipeline = AutoMLPipeline()

# Load and process data
pipeline.load_data("your_data.csv")
pipeline.preprocess_data()

# Train models
pipeline.train_models()

# Generate insights
pipeline.generate_business_intelligence()
```

## 🔧 Configuration

### Environment Variables
- Set your API keys in the `.env` file (use `env_example.txt` as template)
- Configure logging and model parameters in environment variables

### Data Processing Configuration
- Column types are automatically detected and can be manually configured
- Encoders are automatically generated and stored in `data_processing/encoders/`

## 📈 Features in Detail

### Data Processing
- **Intelligent Cleaning**: Automatic handling of missing values, outliers, and data quality issues
- **Smart Encoding**: Categorical variable encoding with preservation of relationships
- **Feature Engineering**: Automated feature creation and selection

### Pattern Discovery
- **Statistical Analysis**: Correlation analysis, ANOVA, Chi-square tests
- **Outlier Detection**: Multiple methods for identifying and handling outliers
- **Time Series Analysis**: Trend detection and seasonal patterns
- **Segment Analysis**: Customer/group segmentation and analysis

### AutoML Pipeline
- **Multiple Algorithms**: GBM, Random Forest, FLAML with automatic selection
- **Hyperparameter Tuning**: Automated optimization of model parameters
- **Cross-Validation**: Robust model evaluation
- **Ensemble Methods**: Combining multiple models for better performance

### Business Intelligence
- **AI-Powered Insights**: Natural language explanations of findings
- **Recommendation Engine**: Actionable business recommendations
- **Visual Reports**: Interactive charts and graphs
- **Executive Summaries**: High-level insights for decision makers

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- 📧 Email: support@automl-analyst.com
- 📖 Documentation: [docs/](docs/)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/automl-analyst/issues)

## 🙏 Acknowledgments

- FLAML for AutoML capabilities
- Streamlit for the web interface
- Together AI for language model integration
- The open-source community for various libraries and tools

---

**Made with ❤️ for data scientists and analysts**
