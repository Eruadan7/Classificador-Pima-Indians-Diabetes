import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from pickle import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from pprint import pprint


# REGRESSÃO LOGÍSTICA


# 1. CARREGAR OS DADOS
dados = pd.read_csv('diabetes.csv', sep=',')

# 2. SEPARAR ATRIBUTOS E CLASSE
dados_atributos = dados.drop('Outcome', axis=1)
dados_classe = dados['Outcome']

# 3. NORMALIZAÇÃO 
scaler = StandardScaler()
dados_atributos_normalizados = scaler.fit_transform(dados_atributos)

# 4. BALANCEAMENTO COM SMOTE
resampler = SMOTE(random_state=42)
atributos_b, classes_b = resampler.fit_resample(dados_atributos_normalizados, dados_classe)

print("Distribuição após SMOTE:")
print(pd.Series(classes_b).value_counts())

# 5. DEFINIR GRADE DE HIPERPARÂMETROS
lr_grid = {
    'C': [0.01, 0.1, 1, 10, 100],           # Regularização (inverso)
    'penalty': ['l1', 'l2', 'elasticnet'],   # Tipo de penalidade
    'solver': ['lbfgs', 'liblinear', 'saga'], # Algoritmo de otimização
    'max_iter': [100, 200, 500]              # Número máximo de iterações
}

# 6. OTIMIZAÇÃO COM RANDOMIZEDSEARCHCV
lr = LogisticRegression(random_state=42)

lr_hyperparameters = RandomizedSearchCV(
    estimator=lr,
    param_distributions=lr_grid,
    n_iter=20,
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

lr_hyperparameters.fit(atributos_b, classes_b)

print("\n=== MELHORES PARÂMETROS REGRESSÃO LOGÍSTICA ===")
pprint(lr_hyperparameters.best_params_)

# 7. INSTANCIAR MODELO OTIMIZADO
lr_otimizado = LogisticRegression(**lr_hyperparameters.best_params_, random_state=42)

# 8. VALIDAÇÃO CRUZADA FINAL (10 FOLDS)
scoring = ['accuracy', 'f1_macro', 'precision', 'recall']

scores_cross = cross_validate(
    lr_otimizado,
    atributos_b, classes_b,
    scoring=scoring,
    n_jobs=-1,
    cv=10,
    verbose=0
)

print("\n=== RESULTADOS DA VALIDAÇÃO CRUZADA (cv=10) ===")
print(f"Acurácia média:  {scores_cross['test_accuracy'].mean():.4f} (+/- {scores_cross['test_accuracy'].std():.4f})")
print(f"Precision média: {scores_cross['test_precision'].mean():.4f}")
print(f"Recall médio:    {scores_cross['test_recall'].mean():.4f}")
print(f"F1-Score médio:  {scores_cross['test_f1_macro'].mean():.4f}")

# 9. TREINAR MODELO FINAL
#diabetes_lr = lr_otimizado.fit(atributos_b, classes_b)

# 10. SALVAR MODELO E O SCALER
# dump(diabetes_lr, open('diabetes_logistic_regression.pkl', 'wb'))
# dump(scaler, open('diabetes_scaler_lr.pkl', 'wb'))