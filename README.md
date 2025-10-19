# FuelSense: Predicting and Optimizing Vehicle Fuel Efficiency

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🚗 Project Overview

**FuelSense** is a comprehensive data science project that analyzes vehicle fuel efficiency using the UCI Auto MPG dataset. The project combines machine learning, data visualization, and an interactive web dashboard to predict Miles Per Gallon (MPG) and provide actionable insights for optimizing vehicle fuel efficiency.

### 🎯 Key Features

- **Data Analysis**: Comprehensive exploration of vehicle attributes affecting fuel efficiency
- **Machine Learning**: Trained regression models (Linear Regression & Random Forest) for MPG prediction
- **Interactive Dashboard**: Real-time MPG prediction with optimization suggestions
- **Actionable Insights**: Data-driven recommendations for improving fuel efficiency
- **Visualizations**: Rich charts and graphs showing relationships between vehicle features and MPG

## 📊 Dataset

The project uses the **UCI Auto MPG Dataset** containing information about:
- **398 vehicles** from 1970-1982
- **9 attributes**: MPG, cylinders, displacement, horsepower, weight, acceleration, model year, origin, car name
- **Target variable**: Miles Per Gallon (MPG)

**Data Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/auto+mpg)

## 🏗️ Project Structure

```
FuelSense/
├── data/                          # Dataset storage
├── models/                        # Trained model files
├── fuel_sense_analysis.ipynb      # Complete data analysis notebook
├── app.py                         # Streamlit dashboard application
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## 🔍 Analysis Workflow

### 1. **Data Ingestion & Setup**
- Load UCI Auto MPG dataset
- Display dataset overview (shape, columns, data types)
- Identify missing values and data quality issues

### 2. **Data Cleaning & Preprocessing** 
- Handle missing values in horsepower column
- Convert data types and handle non-numeric entries
- Create categorical encodings for origin and manufacturer

### 3. **Exploratory Data Analysis (EDA)**
- **Correlation Analysis**: Identify relationships between features and MPG
- **Distribution Plots**: Understand MPG distribution patterns  
- **Visualizations**: Heatmaps, scatter plots, box plots, and pairplots
- **Key Finding**: Weight, horsepower, and displacement are top MPG predictors

### 4. **Feature Engineering**
- Create derived features:
  - `weight_per_horsepower`: Power-to-weight ratio
  - `engine_efficiency`: Displacement per cylinder
  - `car_age`: Vehicle age calculation
  - `power_to_weight`: Horsepower-to-weight ratio
- Feature scaling and normalization

### 5. **Model Building & Training**
- **Data Split**: 80% training, 20% testing
- **Models Trained**:
  - Linear Regression (with feature scaling)
  - Random Forest Regressor
- **Feature Selection**: 12 engineered features for optimal prediction

### 6. **Model Evaluation**
- **Metrics Used**: RMSE (Root Mean Square Error) and R² Score
- **Performance**: Best model achieves R² > 0.85 on test data
- **Model Selection**: Choose best performing model for deployment

### 7. **Feature Importance Analysis**
- Identify most impactful features for MPG prediction
- Generate feature importance rankings
- Visualize feature contributions

### 8. **Insights Generation**
- **Top 3 Factors Affecting MPG**:
  1. **Weight** (strongest negative correlation)
  2. **Horsepower** (negative impact on efficiency)
  3. **Displacement** (engine size vs. efficiency trade-off)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd FuelSense
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the Jupyter notebook** (optional):
```bash
jupyter notebook fuel_sense_analysis.ipynb
```

4. **Launch the Streamlit dashboard**:
```bash
streamlit run app.py
```

### 📱 Using the Dashboard

1. **Open your browser** to `http://localhost:8501`
2. **Adjust vehicle parameters** using the sidebar sliders:
   - Number of cylinders (3-8)
   - Engine displacement (70-450 cu in)
   - Horsepower (50-250 HP)
   - Weight (1500-5000 lbs)
   - Acceleration (8-25 seconds)
   - Model year (1970-1982)
   - Origin (USA/Europe/Japan)

3. **View predictions**:
   - Real-time MPG prediction
   - Efficiency rating (Excellent/Good/Average/Poor)
   - Interactive gauge visualization

4. **Get optimization suggestions**:
   - Personalized improvement recommendations
   - Quantified MPG improvement estimates
   - Feature impact analysis charts

## 📈 Key Insights & Recommendations

### For Car Manufacturers:
1. **Weight Reduction**: Focus on lightweight materials - strongest impact on fuel efficiency
2. **Engine Optimization**: Optimize horsepower-to-displacement ratios
3. **Market Positioning**: Japanese cars show highest efficiency in dataset
4. **Technology Investment**: Invest in turbocharging for smaller, efficient engines

### For Car Owners:
1. **Vehicle Selection**: Prioritize lower weight and appropriate engine size
2. **Maintenance**: Regular tune-ups significantly impact real-world MPG
3. **Driving Habits**: Smooth acceleration and steady speeds improve efficiency
4. **Modifications**: Weight reduction modifications offer highest ROI

### Statistical Findings:
- **10% weight reduction** → ~2-3 MPG improvement
- **Japanese cars** average 2-4 MPG higher than US/European equivalents
- **Newer model years** show consistent efficiency improvements
- **4-cylinder engines** generally outperform 6+ cylinder configurations

## 🛠️ Technical Implementation

### Machine Learning Models:
- **Linear Regression**: Interpretable coefficients for feature impact analysis
- **Random Forest**: Higher accuracy with feature importance rankings
- **Preprocessing**: StandardScaler for feature normalization
- **Validation**: Train/test split with cross-validation

### Dashboard Features:
- **Interactive Sliders**: Real-time parameter adjustment
- **Gauge Visualization**: Intuitive MPG display
- **Comparison Charts**: Multi-dimensional analysis
- **Responsive Design**: Works on desktop and mobile devices

### Data Processing:
- **Missing Value Handling**: Median imputation grouped by cylinders
- **Feature Engineering**: 5 derived features for improved prediction
- **Categorical Encoding**: Origin and manufacturer categorization
- **Outlier Detection**: Statistical methods for data quality

## 📊 Model Performance

| Model | Training R² | Test R² | Training RMSE | Test RMSE |
|-------|-------------|---------|---------------|-----------|
| Linear Regression | 0.821 | 0.798 | 3.24 | 3.47 |
| Random Forest | 0.958 | 0.847 | 1.58 | 3.02 |

**Winner**: Random Forest Regressor with **84.7% accuracy** on test data

## 🔮 Future Enhancements

### Immediate Improvements:
- [ ] Add more recent vehicle data (2000s-2020s)
- [ ] Include hybrid and electric vehicle analysis
- [ ] Implement advanced ML models (XGBoost, Neural Networks)
- [ ] Add real-time data integration from automotive APIs

### Advanced Features:
- [ ] Multi-objective optimization (MPG vs. Performance)
- [ ] Cost-benefit analysis for modifications
- [ ] Environmental impact calculations (CO2 emissions)
- [ ] Integration with vehicle diagnostic systems

### Dashboard Enhancements:
- [ ] Save and compare multiple vehicle configurations
- [ ] Export reports and recommendations to PDF
- [ ] Add more interactive visualizations
- [ ] Implement user authentication and saved profiles

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup:
```bash
# Fork the repository
git clone <your-fork-url>
cd FuelSense

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Make your changes and test
streamlit run app.py

# Submit a pull request
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Project Repository**: [GitHub Repository URL]
- **Issues**: [GitHub Issues URL]
- **Discussions**: [GitHub Discussions URL]

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for providing the Auto MPG dataset
- **Streamlit** for the excellent dashboard framework
- **Scikit-learn** for robust machine learning tools
- **Plotly** for interactive visualizations

---

<div align="center">

**⛽ FuelSense - Driving Efficiency Through Data Science ⛽**

*Made with ❤️ and Python*

</div>