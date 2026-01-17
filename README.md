# spt-ml-property-predictor
Machine Learning model (Random Forest) for predicting soil engineering properties from SPT data - Data-driven site characterization

# SPT-Based Soil Property Predictor Using Machine Learning

A Random Forest machine learning model that predicts critical soil engineering properties from Standard Penetration Test (SPT) N-values. This automates the traditional correlation-based approach used in geotechnical site characterization.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-Random%20Forest-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## 🎯 Project Objective

Replace manual lookup tables and empirical correlations with a data-driven machine learning approach for predicting:
- **Friction Angle (φ)** - Critical for bearing capacity calculations
- **Relative Density (Dr)** - Sand classification and settlement estimation
- **Bearing Capacity (qult)** - Foundation design parameter
- **Elastic Modulus (E)** - Settlement analysis

## 🧠 Why Machine Learning?

Traditional geotechnical practice uses **empirical correlations** (Peck, Bowles, IS codes) which:
- ❌ Are region-specific and may not apply universally
- ❌ Don't capture complex non-linear relationships
- ❌ Can't learn from new data

**Machine Learning approach:**
- ✅ Learns patterns from actual data
- ✅ Captures complex relationships between features
- ✅ Improves with more data
- ✅ Provides uncertainty estimates

## 📊 Dataset

**Source:** Indian SPT field data (NGM-MSM dataset)

**Features:**
- SPT N-value (corrected)
- Depth
- Soil type
- [Additional features based on available data]

**Target Variables:**
- Friction angle (degrees)
- Relative density (%)
- Bearing capacity (kPa)
- Elastic modulus (MPa)

## 🛠️ Methodology

### 1. Data Preparation
- Load SPT data from Excel
- Apply standard corrections (N60)
- Generate target variables using established correlations
- Split into training (80%) and testing (20%) sets

### 2. Feature Engineering
- Corrected N-values
- Depth-based features
- Soil type encoding
- Derived parameters

### 3. Model Training
- Algorithm: **Random Forest Regressor**
- Hyperparameter tuning using GridSearchCV
- Cross-validation (5-fold)

### 4. Evaluation
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Comparison with traditional correlations

### 5. Deployment
- Save trained model (.pkl file)
- Create prediction interface
- Visualize results

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/Femi-Blessing-Geotech/spt-ml-property-predictor.git

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook
```

## 💻 Usage Example
```python
from src.predictor import SPTPropertyPredictor
import pandas as pd

# Initialize predictor
predictor = SPTPropertyPredictor()

# Load and train model
predictor.train(data_path='data/IndiaNGMMSM.xlsx')

# Predict for new SPT data
new_data = pd.DataFrame({
    'N_corrected': [25, 30, 15],
    'Depth': [2.0, 5.0, 8.0],
    'Soil_Type': ['Sand', 'Sand', 'Clay']
})

predictions = predictor.predict(new_data)
print(predictions)
# Output:
# Friction_Angle: [35.2, 37.8, 28.5]
# Bearing_Capacity: [450, 520, 280]
```

## 📁 Project Structure
```
spt-ml-property-predictor/
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   ├── raw/
│   │   └── IndiaNGMMSM.xlsx
│   └── processed/
│       └── prepared_data.csv
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation_results.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_trainer.py
│   └── predictor.py
├── models/
│   ├── rf_friction_angle.pkl
│   ├── rf_bearing_capacity.pkl
│   └── model_metrics.json
├── results/
│   ├── model_performance.png
│   ├── feature_importance.png
│   └── predictions_vs_actual.png
└── app/
    └── streamlit_app.py (Interactive web app - optional)
```

## 📈 Expected Results

**Model Performance Targets:**
- R² Score: > 0.85
- MAE: < 3° for friction angle
- Comparison shows ML outperforms single correlations

## 🔬 Technical Details

**Libraries:**
- scikit-learn (RandomForestRegressor)
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib/seaborn (visualization)
- joblib (model persistence)

**Model Architecture:**
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)
```

## 📚 References

**Traditional Correlations Used for Validation:**
- Peck et al. (1974) - Friction angle from SPT
- Bowles (1996) - Bearing capacity correlations
- IS 6403:1981 - Indian Standard correlations
- Terzaghi & Peck (1967) - Foundation design

**Machine Learning:**
- Breiman (2001) - Random Forests
- Scikit-learn documentation
- Recent papers on ML in geotechnical engineering

## 🎯 Learning Outcomes

By completing this project, you will:
- ✅ Build end-to-end ML pipeline for geotechnical data
- ✅ Apply Random Forest algorithm to real-world problem
- ✅ Evaluate model performance with proper metrics
- ✅ Compare ML approach with traditional methods
- ✅ Deploy trained model for predictions
- ✅ Create professional visualizations

## 🔄 Development Roadmap

**Phase 1: Data Preparation (Days 1-3)**
- [x] Repository setup
- [ ] Load and clean SPT data
- [ ] Generate target variables using correlations
- [ ] Train/test split

**Phase 2: Model Development (Days 4-7)**
- [ ] Feature engineering
- [ ] Train Random Forest model
- [ ] Hyperparameter tuning
- [ ] Cross-validation

**Phase 3: Evaluation (Days 8-10)**
- [ ] Calculate performance metrics
- [ ] Compare with traditional correlations
- [ ] Feature importance analysis
- [ ] Error analysis

**Phase 4: Deployment (Days 11-14)**
- [ ] Save trained model
- [ ] Create prediction interface
- [ ] Generate visualizations
- [ ] Document results

## 🌟 Future Enhancements

- Add more ML algorithms (XGBoost, Neural Networks)
- Ensemble methods combining multiple models
- Uncertainty quantification (prediction intervals)
- Web interface for easy prediction
- Integration with CPT data
- Spatial prediction across site

## 👤 Author

**Femi Blessing**
- GitHub: [@Femi-Blessing-Geotech](https://github.com/Femi-Blessing-Geotech)
- Focus: Data-driven geotechnical site characterization
- Vision: Pioneering ML applications in offshore geotechnics

## 🏆 Impact

This project demonstrates:
- **Technical Skills:** Python, ML, geotechnical engineering
- **Innovation:** Modern approach to traditional problem
- **Practical Value:** Directly applicable to foundation design
- **Research Potential:** Foundation for PhD-level work

---

⭐ **Star this repo if you find it useful!**

📧 **Questions or collaboration?** Open an issue or reach out!

🔗 **Part of my portfolio:** [github.com/Femi-Blessing-Geotech](https://github.com/Femi-Blessing-Geotech)
```

Commit this README.

---

## **STEP 3: THE ACTUAL ML CODE** (THIS IS WHAT YOU WANT!)

### **Create requirements.txt:**
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
openpyxl>=3.0.0
jupyter>=1.0.0
joblib>=1.2.0
