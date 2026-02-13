# Urban Sustainability Prediction - Your Assignment Guide
## Customized for Your Dataset

---

## 🎯 **What You're Predicting**

**Target**: `urban_sustainability_score` (0 to 1 scale)

You're helping an AECO firm predict **how sustainable an urban area will be** based on planning characteristics like building density, green space, transport access, and more.

---

## 📊 **Your Dataset: Urban Planning Dataset**

**Dataset Overview**:
- **3,476 urban areas** (rows)
- **15 features** (predictors)
- **1 target** (urban_sustainability_score)
- **Perfect for AECO because**:
  - Real estate developers need to assess area sustainability before investing
  - Urban planners want to optimize city designs for sustainability
  - Construction firms need to understand which factors drive green building certifications
  - Government agencies need data-driven urban development strategies

**Features in your dataset**:
1. `building_density` - How densely buildings are packed
2. `road_connectivity` - Quality of road network
3. `public_transport_access` - Access to buses/trains/metro
4. `air_quality_index` - Air pollution levels
5. `green_cover_percentage` - Parks, trees, green spaces
6. `carbon_footprint` - CO2 emissions
7. `population_density` - People per square km
8. `crime_rate` - Safety indicator
9. `avg_income` - Economic indicator
10. `renewable_energy_usage` - Solar/wind/clean energy use
11. `disaster_risk_index` - Flood/earthquake/fire risk
12. `land_use_type_Commercial` - Is it commercial area? (1=yes, 0=no)
13. `land_use_type_Green Space` - Is it green space? (1=yes, 0=no)
14. `land_use_type_Industrial` - Is it industrial? (1=yes, 0=no)
15. `land_use_type_Residential` - Is it residential? (1=yes, 0=no)

**Good news**: No missing values! Data is clean and ready to use.

---

## 🚀 **Your Complete Notebook Code**

### **Step 1: Import Libraries & Load Data**

```python
# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Set display options
pd.set_option('display.max_columns', None)
np.random.seed(42)

# Load your dataset
df = pd.read_csv('urban_planning_dataset.csv')

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
df.head()
```

---

### **Step 2: Explore the Data**

```python
# Basic info
print("Dataset Information:")
print(df.info())

print("\n" + "="*60)
print("Statistical Summary:")
print(df.describe())

print("\n" + "="*60)
print("Missing Values:")
print(df.isnull().sum())

print("\n" + "="*60)
print("Target Variable Distribution:")
print(f"Mean Sustainability Score: {df['urban_sustainability_score'].mean():.3f}")
print(f"Min: {df['urban_sustainability_score'].min():.3f}")
print(f"Max: {df['urban_sustainability_score'].max():.3f}")
```

```python
# Visualize target distribution
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(df['urban_sustainability_score'], bins=30, edgecolor='black')
plt.xlabel('Urban Sustainability Score')
plt.ylabel('Frequency')
plt.title('Distribution of Sustainability Scores')

plt.subplot(1, 2, 2)
plt.boxplot(df['urban_sustainability_score'])
plt.ylabel('Urban Sustainability Score')
plt.title('Box Plot of Target Variable')

plt.tight_layout()
plt.show()
```

---

### **Step 3: Prepare Features and Target**

```python
# Define target variable
target = 'urban_sustainability_score'
y = df[target]

# Define features (all columns except target)
feature_cols = [col for col in df.columns if col != target]
X = df[feature_cols]

print(f"Number of features: {len(feature_cols)}")
print(f"Features: {feature_cols}")
print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
```

```python
# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
```

---

### **Step 4: Model A - Linear Regression**

**Why Linear Regression?**
- Simple and fast
- Easy to interpret coefficients
- Good baseline for continuous predictions
- Shows which features have linear relationships with sustainability

```python
# Train Linear Regression
print("="*60)
print("MODEL A: LINEAR REGRESSION")
print("="*60)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Make predictions
y_pred_lr = model_lr.predict(X_test)

# Calculate metrics
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print(f"\nLinear Regression Results:")
print(f"MAE:  {mae_lr:.4f}")
print(f"RMSE: {rmse_lr:.4f}")
print(f"R² Score: {r2_lr:.4f}")
print(f"\nInterpretation: The model explains {r2_lr*100:.1f}% of sustainability variance")
```

```python
# Visualize predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Sustainability Score')
plt.ylabel('Predicted Sustainability Score')
plt.title('Linear Regression: Actual vs Predicted')
plt.tight_layout()
plt.show()
```

---

### **Step 5: Model B - Random Forest**

**Why Random Forest?**
- Captures complex, non-linear relationships
- Handles interactions between features (e.g., high green space + low carbon = high sustainability)
- More robust and flexible than linear models
- Can identify feature importance

```python
# Train Random Forest
print("="*60)
print("MODEL B: RANDOM FOREST")
print("="*60)

model_rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model_rf.fit(X_train, y_train)

# Make predictions
y_pred_rf = model_rf.predict(X_test)

# Calculate metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"\nRandom Forest Results:")
print(f"MAE:  {mae_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R² Score: {r2_rf:.4f}")
print(f"\nInterpretation: The model explains {r2_rf*100:.1f}% of sustainability variance")
```

```python
# Visualize predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Sustainability Score')
plt.ylabel('Predicted Sustainability Score')
plt.title('Random Forest: Actual vs Predicted')
plt.tight_layout()
plt.show()
```

---

### **Step 6: Compare Models**

```python
# Create comparison table
print("="*60)
print("MODEL COMPARISON")
print("="*60)

comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MAE': [mae_lr, mae_rf],
    'RMSE': [rmse_lr, rmse_rf],
    'R² Score': [r2_lr, r2_rf]
})

print(comparison)

# Determine winner
winner = 'Random Forest' if r2_rf > r2_lr else 'Linear Regression'
print(f"\n🏆 WINNER: {winner}")
print(f"Reason: Higher R² score ({max(r2_lr, r2_rf):.4f})")
```

```python
# Side-by-side visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear Regression plot
axes[0].scatter(y_test, y_pred_lr, alpha=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Score')
axes[0].set_ylabel('Predicted Score')
axes[0].set_title(f'Linear Regression\nR² = {r2_lr:.3f}')

# Random Forest plot
axes[1].scatter(y_test, y_pred_rf, alpha=0.5, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Score')
axes[1].set_ylabel('Predicted Score')
axes[1].set_title(f'Random Forest\nR² = {r2_rf:.3f}')

plt.tight_layout()
plt.show()
```

---

### **Step 7: Feature Importance (Random Forest Only)**

```python
# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model_rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("="*60)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*60)
print(feature_importance.head(10))

# Visualize top 10 features
plt.figure(figsize=(10, 6))
top_10 = feature_importance.head(10)
plt.barh(top_10['Feature'], top_10['Importance'])
plt.xlabel('Importance Score')
plt.title('Top 10 Features for Predicting Urban Sustainability')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

---

### **Step 8: Optimize the Winner (Random Forest)**

```python
print("="*60)
print("OPTIMIZING RANDOM FOREST")
print("="*60)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='r2',
    verbose=1,
    n_jobs=-1
)

print("Running grid search... (this may take a few minutes)")
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_
print(f"\nBest Parameters: {grid_search.best_params_}")

# Evaluate optimized model
y_pred_optimized = best_model.predict(X_test)
mae_opt = mean_absolute_error(y_test, y_pred_optimized)
rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_optimized))
r2_opt = r2_score(y_test, y_pred_optimized)

print(f"\nOptimized Random Forest Results:")
print(f"MAE:  {mae_opt:.4f}")
print(f"RMSE: {rmse_opt:.4f}")
print(f"R² Score: {r2_opt:.4f}")

# Compare before and after optimization
improvement = pd.DataFrame({
    'Version': ['Original RF', 'Optimized RF'],
    'MAE': [mae_rf, mae_opt],
    'RMSE': [rmse_rf, rmse_opt],
    'R² Score': [r2_rf, r2_opt]
})

print("\n" + "="*60)
print("BEFORE vs AFTER OPTIMIZATION")
print("="*60)
print(improvement)
```

---

### **Step 9: Final Summary**

```python
print("="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

final_results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Optimized Random Forest'],
    'MAE': [mae_lr, mae_rf, mae_opt],
    'RMSE': [rmse_lr, rmse_rf, rmse_opt],
    'R² Score': [r2_lr, r2_rf, r2_opt]
})

print(final_results)
print("\n" + "="*60)
print(f"🏆 BEST MODEL: Optimized Random Forest")
print(f"📊 Final R² Score: {r2_opt:.4f} ({r2_opt*100:.1f}% variance explained)")
print(f"📉 Mean Absolute Error: {mae_opt:.4f} (average prediction error)")
print("="*60)
```

---

## 📝 **Your Report Template**

Use this structure for your PDF report:

---

### **ANALYST REPORT: Urban Sustainability Prediction Engine**

**Prepared for**: [Company Name] AECO Board  
**Date**: [Today's Date]  
**Analyst**: [Your Name]

---

#### **1. EXECUTIVE SUMMARY**

This report presents a predictive engine to forecast urban sustainability scores using historical urban planning data. Two machine learning models were compared, with Random Forest emerging as the superior choice, achieving an R² score of [X.XXX] after optimization.

---

#### **2. DATASET SELECTION**

**Dataset**: Urban Planning Dataset  
**Source**: Kaggle  
**Size**: 3,476 urban areas with 15 planning characteristics

**Why This Dataset?**

This dataset is highly relevant to AECO because:
- **Real Estate Development**: Developers can assess sustainability before investing in new areas
- **Urban Planning**: City planners can identify which factors drive sustainable outcomes
- **Construction Strategy**: Firms can design projects that contribute to higher area sustainability
- **Green Certification**: Understanding sustainability predictors helps achieve LEED/BREEAM certifications
- **Investment Risk**: Sustainability scores correlate with long-term property values and regulatory compliance

The dataset includes critical AECO factors: building density, green space coverage, transport access, carbon emissions, and land use types.

---

#### **3. METHODOLOGY**

**Machine Learning Models Selected**:

**Model A: Linear Regression**
- **Rationale**: Simple, interpretable baseline model
- **Strengths**: Fast training, clear coefficient interpretation
- **Use Case**: When we need to understand direct relationships between features and sustainability

**Model B: Random Forest**
- **Rationale**: Captures complex, non-linear relationships and feature interactions
- **Strengths**: Handles complex patterns, resistant to overfitting, provides feature importance
- **Use Case**: When sustainability depends on combinations of factors (e.g., high green space + low carbon footprint)

**Data Split**: 80% training (2,780 areas) / 20% testing (696 areas)

---

#### **4. RESULTS**

| Model | MAE | RMSE | R² Score | Variance Explained |
|-------|-----|------|----------|-------------------|
| Linear Regression | [X.XXXX] | [X.XXXX] | [X.XXXX] | [XX.X%] |
| Random Forest | [X.XXXX] | [X.XXXX] | [X.XXXX] | [XX.X%] |
| **Optimized RF** | **[X.XXXX]** | **[X.XXXX]** | **[X.XXXX]** | **[XX.X%]** |

**Key Findings**:
- Random Forest outperformed Linear Regression by [X]%
- Optimization improved performance by an additional [X]%
- The model is typically off by only [X.XX] points (on a 0-1 scale)

---

#### **5. FEATURE IMPORTANCE**

**Top 5 Predictors of Urban Sustainability**:
1. [Feature name] - [Brief explanation]
2. [Feature name] - [Brief explanation]
3. [Feature name] - [Brief explanation]
4. [Feature name] - [Brief explanation]
5. [Feature name] - [Brief explanation]

---

#### **6. RECOMMENDATION**

**Recommended Model: Optimized Random Forest**

**Why?**
- **Accuracy**: Achieves R² of [X.XXX], explaining [XX%] of sustainability variance
- **Precision**: Mean Absolute Error of [X.XX] means predictions are highly reliable
- **Robustness**: Handles complex interactions between urban planning factors
- **Actionable Insights**: Feature importance reveals which factors to prioritize

**Business Impact**:
- Predict sustainability scores for proposed development sites
- Identify areas requiring intervention to improve sustainability
- Support data-driven urban development strategies
- Benchmark new projects against predicted outcomes

---

#### **7. IMPLEMENTATION ROADMAP**

**Phase 1 (Immediate)**:
- Deploy model to assess current portfolio properties
- Create sustainability scoring dashboard

**Phase 2 (3-6 months)**:
- Integrate with project planning workflow
- Train team on interpreting predictions

**Phase 3 (6-12 months)**:
- Expand model with additional data sources
- Develop what-if scenario planning tool

---

#### **8. LIMITATIONS & NEXT STEPS**

**Current Limitations**:
- Model trained on historical data; may not capture emerging trends
- Doesn't account for future policy changes or climate factors
- Geographic specificity unknown

**Recommended Next Steps**:
- Collect company-specific project data to retrain model
- Add temporal features (year, seasonality)
- Develop separate models for different climate zones or regions
- Implement continuous model monitoring and retraining

---

#### **CONCLUSION**

The Optimized Random Forest model provides [Company Name] with a robust, accurate tool to predict urban sustainability outcomes. With an R² score of [X.XXX], this "Predictive Engine" can guide strategic decisions in real estate investment, urban planning, and sustainable construction projects.

---

## ✅ **Your Submission Checklist**

- [ ] Jupyter Notebook with all 9 code steps
- [ ] All code cells executed with output visible
- [ ] Visualizations included (at least 4)
- [ ] Comments explaining each section
- [ ] Report answers all 4 required questions:
  - [ ] Why did you choose this dataset? ✓
  - [ ] Why did you choose these two algorithms? ✓
  - [ ] Which model is best? ✓
  - [ ] What was the final accuracy/error rate? ✓
- [ ] Report saved as PDF
- [ ] Business-friendly language (no excessive jargon)
- [ ] Clear recommendation with justification

---

## 💡 **Key Metrics Explained**

**MAE (Mean Absolute Error)**:
- Average prediction error
- Example: MAE of 0.05 means predictions are off by ±0.05 on average (on 0-1 scale)
- Lower is better

**RMSE (Root Mean Squared Error)**:
- Similar to MAE but penalizes large errors more
- Useful for identifying if model makes occasional big mistakes
- Lower is better

**R² Score (Coefficient of Determination)**:
- Percentage of variance explained (0 to 1)
- Example: 0.85 means model explains 85% of why areas have different sustainability scores
- Higher is better
- > 0.70 is generally considered good for this type of problem

---

## 🎓 **Pro Tips for Success**

1. **Add markdown cells** in Jupyter between code sections explaining what you're doing
2. **Include print statements** showing intermediate results
3. **Label all visualizations** clearly with titles and axis labels
4. **Write your report first in bullet points**, then expand to full sentences
5. **Have someone non-technical read your report** - if they understand it, you're good!

---

**You're all set! Follow this guide step-by-step and you'll have an excellent submission.**
