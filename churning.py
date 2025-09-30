import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, precision_score, 
                           recall_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import json
import joblib

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("BANK CUSTOMER CHURN PREDICTION - HYPERPARAMETER TUNING")
print("=" * 60)

# Load data and prepare features
df_processed = pd.read_csv("C:\\Users\\Admin\\Desktop\\churning system\\bank_churn_processed.csv")
feature_columns = ['credit_score', 'age', 'tenure', 'balance', 'estimated_salary', 
                  'products_number', 'country_encoded', 'gender_encoded', 
                  'credit_card', 'active_member', 'balance_salary_ratio', 'is_high_balance']

X = df_processed[feature_columns]
y = df_processed['churn']

# Train-test split (consistent with previous phases)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Load previous results
try:
    with open('model_results.json', 'r') as f:
        previous_results = json.load(f)
    print(f"\nPrevious Best Model: {previous_results['best_model']}")
    print(f"Previous Best AUC: {previous_results['best_auc']:.4f}")
except FileNotFoundError:
    print("Previous results not found, starting fresh...")
    previous_results = {'best_auc': 0.0, 'best_model': 'None'}

# 1. Simplified Hyperparameter Tuning
print("\n1. SIMPLIFIED HYPERPARAMETER TUNING")
print("-" * 40)

# Use simpler cross-validation to reduce memory usage
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced from 5 to 3

# Simplified parameter grids to avoid memory issues
param_grids = {}

# Random Forest - Simplified
param_grids['Random Forest'] = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# XGBoost - Simplified
param_grids['XGBoost'] = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 0.9]
}

# Gradient Boosting - Simplified
param_grids['Gradient Boosting'] = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5]
}

print("✓ Simplified parameter grids defined")
print("✓ 3-fold stratified cross-validation (reduced for memory)")

# 2. Sequential Tuning (No Parallel Processing)
print("\n2. SEQUENTIAL TUNING (MEMORY-EFFICIENT)")
print("-" * 40)

tuned_models = {}
tuning_results = {}

for model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
    print(f"\n🔧 Tuning {model_name}...")
    
    # Initialize model
    if model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=1)  # Single job
    elif model_name == 'XGBoost':
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, n_jobs=1)
    else:  # Gradient Boosting
        model = GradientBoostingClassifier(random_state=42)
    
    # Reduced Randomized Search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grids[model_name],
        n_iter=10,  # Reduced from 50 to 10
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=1,  # Single job to avoid memory issues
        random_state=42,
        verbose=1
    )
    
    # Fit the model
    try:
        random_search.fit(X_train, y_train)
        
        # Store results
        tuned_models[model_name] = random_search.best_estimator_
        tuning_results[model_name] = {
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_,
            'best_estimator': random_search.best_estimator_
        }
        
        print(f"✓ Best CV Score: {random_search.best_score_:.4f}")
        print(f"✓ Best parameters: {random_search.best_params_}")
        
    except Exception as e:
        print(f"❌ Error tuning {model_name}: {e}")
        print("Using default parameters...")
        # Use default model if tuning fails
        model.fit(X_train, y_train)
        tuned_models[model_name] = model
        tuning_results[model_name] = {
            'best_score': 0.0,
            'best_params': 'default',
            'best_estimator': model
        }

# 3. Manual Fine-tuning (No Grid Search)
print("\n3. MANUAL FINE-TUNING")
print("-" * 40)

final_models = {}

for model_name in ['Random Forest', 'XGBoost']:
    if model_name in tuning_results and tuning_results[model_name]['best_params'] != 'default':
        print(f"\n🎯 Applying best parameters for {model_name}...")
        final_models[model_name] = tuned_models[model_name]
    else:
        print(f"\n🎯 Using default {model_name}...")
        if model_name == 'Random Forest':
            final_models[model_name] = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                class_weight='balanced',
                n_jobs=1
            )
        else:  # XGBoost
            final_models[model_name] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=1
            )
        final_models[model_name].fit(X_train, y_train)

# 4. Ensemble Method
print("\n4. ENSEMBLE METHOD")
print("-" * 40)

ensemble_models = {}

try:
    # Voting Classifier with available models
    available_models = []
    for name, model in final_models.items():
        available_models.append((name.lower(), model))
    
    if available_models:
        voting_clf = VotingClassifier(
            estimators=available_models,
            voting='soft',
            n_jobs=1
        )
        
        voting_clf.fit(X_train, y_train)
        ensemble_models['Voting Classifier'] = voting_clf
        print("✓ Voting Classifier trained")
    else:
        print("❌ No models available for ensemble")
except Exception as e:
    print(f"❌ Error creating ensemble: {e}")

# 5. Performance Evaluation
print("\n5. PERFORMANCE EVALUATION")
print("-" * 40)

# Combine all models for evaluation
all_models = {}
all_models.update(final_models)
all_models.update(ensemble_models)

performance_results = {}

for model_name, model in all_models.items():
    try:
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Simple cross-validation (1 fold to save time/memory)
        cv_score = cross_val_score(model, X_train, y_train, cv=2, scoring='roc_auc', n_jobs=1).mean()
        
        performance_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_score
        }
        
        print(f"\n{model_name}:")
        print(f"  Test ROC-AUC: {roc_auc:.4f}")
        print(f"  CV ROC-AUC: {cv_score:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {e}")

# 6. Results Comparison
print("\n6. RESULTS COMPARISON")
print("-" * 40)

if performance_results:
    # Create comparison dataframe
    comparison_data = []
    for model_name, metrics in performance_results.items():
        comparison_data.append({
            'Model': model_name,
            'Type': 'Tuned/Ensemble',
            'Test_AUC': metrics['roc_auc'],
            'CV_AUC': metrics['cv_mean'],
            'F1_Score': metrics['f1_score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_AUC', ascending=False)
    
    print("\nModel Comparison (Sorted by Test AUC):")
    print("=" * 60)
    print(f"{'Model':<20} {'Test AUC':<10} {'CV AUC':<10} {'F1-Score':<10}")
    print("-" * 60)
    for _, row in comparison_df.iterrows():
        print(f"{row['Model']:<20} {row['Test_AUC']:.4f}    {row['CV_AUC']:.4f}    {row['F1_Score']:.4f}")
    
    # 7. Visualization
    print("\n7. PERFORMANCE VISUALIZATION")
    print("-" * 40)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # AUC Comparison
    models_plot = comparison_df['Model']
    test_auc = comparison_df['Test_AUC']
    
    bars = axes[0].bar(models_plot, test_auc, color='skyblue', alpha=0.8)
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('Test AUC Score')
    axes[0].set_title('Model Comparison - Test AUC')
    axes[0].set_xticklabels(models_plot, rotation=45, ha='right')
    axes[0].set_ylim(0.7, 0.9)
    axes[0].grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, value in zip(bars, test_auc):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{value:.3f}', ha='center', va='bottom')
    
    # F1-Score Comparison
    f1_scores = comparison_df['F1_Score']
    bars = axes[1].bar(models_plot, f1_scores, color='lightgreen', alpha=0.8)
    axes[1].set_xlabel('Models')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title('Model Comparison - F1-Score')
    axes[1].set_xticklabels(models_plot, rotation=45, ha='right')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, value in zip(bars, f1_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # 8. Best Model Analysis
    print("\n8. BEST MODEL ANALYSIS")
    print("-" * 40)
    
    best_model_name = comparison_df.iloc[0]['Model']
    best_model_auc = comparison_df.iloc[0]['Test_AUC']
    best_model = all_models[best_model_name]
    
    print(f"🎯 BEST MODEL: {best_model_name}")
    print(f"   - Test ROC-AUC: {best_model_auc:.4f}")
    
    # Detailed performance
    y_pred_best = best_model.predict(X_test)
    y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]
    
    print(f"\n📊 DETAILED PERFORMANCE:")
    print(classification_report(y_test, y_pred_best, target_names=['Retained', 'Churned']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Predicted Retained', 'Predicted Churned'],
               yticklabels=['Actual Retained', 'Actual Churned'])
    plt.title(f'Confusion Matrix - Best Model ({best_model_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # 9. Feature Importance
    print("\n9. FEATURE IMPORTANCE")
    print("-" * 40)
    
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print("=" * 40)
        for i, row in feature_importance_df.head(10).iterrows():
            print(f"{row['Feature']:<25}: {row['Importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        top_features = feature_importance_df.head(10)
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.xlabel('Importance')
        plt.title(f'Top 10 Feature Importance - {best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    # 10. Model Saving
    print("\n10. MODEL SAVING")
    print("-" * 40)
    
    # Save the best model
    model_filename = f'best_churn_model_{best_model_name.replace(" ", "_")}.pkl'
    joblib.dump(best_model, model_filename)
    
    # Save feature info
    feature_info = {
        'feature_columns': feature_columns,
        'numerical_features': ['credit_score', 'age', 'tenure', 'balance', 'estimated_salary', 'balance_salary_ratio'],
        'categorical_features': ['products_number', 'country_encoded', 'gender_encoded', 'credit_card', 'active_member', 'is_high_balance']
    }
    
    with open('feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("✅ Model artifacts saved:")
    print(f"   - Best model: {model_filename}")
    print(f"   - Feature info: feature_info.json")
    
    # 11. Business Impact
    print("\n11. BUSINESS IMPACT SUMMARY")
    print("-" * 40)
    
    improvement = best_model_auc - previous_results['best_auc']
    total_customers = len(X)
    churn_rate = y.mean()
    
    print(f"📈 PERFORMANCE:")
    print(f"   - Current AUC: {best_model_auc:.4f}")
    print(f"   - Previous Best: {previous_results['best_auc']:.4f}")
    print(f"   - Improvement: {improvement:+.4f}")
    
    print(f"\n🎯 MODEL CAPABILITIES:")
    print(f"   - Recall: {recall_score(y_test, y_pred_best):.1%} of churners identified")
    print(f"   - Precision: {precision_score(y_test, y_pred_best):.1%} of predictions correct")
    print(f"   - Accuracy: {accuracy_score(y_test, y_pred_best):.1%} overall")
    
    print(f"\n💼 BUSINESS VALUE:")
    print(f"   - Total customers analyzed: {total_customers:,}")
    print(f"   - Churn rate: {churn_rate:.1%}")
    print(f"   - Model ready for customer retention strategies")

else:
    print("❌ No models were successfully evaluated")

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING COMPLETED!")
print("=" * 60)
