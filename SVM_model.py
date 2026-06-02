import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from pickle import dump
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from pprint import pprint

# SUPPORT VECTOR MACHINE

# 1. CARREGAR OS DADOS
dados = pd.read_csv('diabetes.csv', sep=',')

# 2. SEPARAR ATRIBUTOS E CLASSE
dados_atributos = dados.drop('Outcome', axis=1)
dados_classe = dados['Outcome']

# 3. NORMALIZAÇÃO
# SVM é sensível à escala
scaler = StandardScaler()
dados_atributos_normalizados = scaler.fit_transform(dados_atributos)

# 4. BALANCEAMENTO COM SMOTE
resampler = SMOTE(random_state=42)
atributos_b, classes_b = resampler.fit_resample(dados_atributos_normalizados, dados_classe)

print("Distribuição após SMOTE:")
print(pd.Series(classes_b).value_counts())

# 5. DEFINIR GRADE DE HIPERPARÂMETROS para SVM
svm_grid = {
    'C': [0.1, 1, 10, 100],                    # Regularização
    'kernel': ['linear', 'rbf', 'poly'],       # Tipo de kernel
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Coeficiente do kernel
    'degree': [2, 3, 4]                        # Apenas para kernel 'poly'
}

# 6. OTIMIZAÇÃO COM RANDOMIZEDSEARCHCV
svm = SVC(random_state=42)

svm_hyperparameters = RandomizedSearchCV(
    estimator=svm,
    param_distributions=svm_grid,
    n_iter=20,
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

svm_hyperparameters.fit(atributos_b, classes_b)

print("\n=== MELHORES PARÂMETROS SVM ===")
pprint(svm_hyperparameters.best_params_)

# 7. INSTANCIAR MODELO OTIMIZADO
svm_otimizado = SVC(**svm_hyperparameters.best_params_, random_state=42)

# 8. VALIDAÇÃO CRUZADA FINAL (10 FOLDS)
scoring = ['accuracy', 'f1_macro', 'precision', 'recall']

scores_cross = cross_validate(
    svm_otimizado,
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
#diabetes_svm = svm_otimizado.fit(atributos_b, classes_b)

# 10. SALVAR MODELO E O SCALER
# dump(diabetes_svm, open('diabetes_svm.pkl', 'wb'))
# dump(scaler, open('diabetes_scaler.pkl', 'wb'))