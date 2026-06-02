import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from pickle import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from pprint import pprint

# 1. CARREGAR OS DADOS
dados = pd.read_csv('diabetes.csv', sep=',')

# 2. SEPARAR ATRIBUTOS E CLASSE (antes da codificação)
dados_atributos = dados.drop('Outcome', axis=1)
dados_classe = dados['Outcome']

# 3. BALANCEAMENTO COM SMOTE
resampler = SMOTE(random_state=42)
atributos_b, classes_b = resampler.fit_resample(dados_atributos, dados_classe)

print("Distribuição após SMOTE:")
print(pd.Series(classes_b).value_counts())


# 4. DEFINIR GRADE DE HIPERPARÂMETROS
rf_grid = {
    'n_estimators': [int(x) for x in np.linspace(10, 100, num=10)],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 10],
    'max_depth': [int(x) for x in np.linspace(10, 100, num=20)],
    'max_features': ['sqrt', 'log2']
}


# 5. OTIMIZAÇÃO COM RANDOMIZEDSEARCHCV
rf = RandomForestClassifier(random_state=42)

rf_hyperparameters = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_grid,
    n_iter=10,
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

rf_hyperparameters.fit(atributos_b, classes_b)

print("\n=== MELHORES PARÂMETROS ===")
pprint(rf_hyperparameters.best_params_)


# 6. INSTANCIAR MODELO OTIMIZADO
rf_otimizado = RandomForestClassifier(**rf_hyperparameters.best_params_, random_state=42)


# 7. VALIDAÇÃO CRUZADA FINAL (10 FOLDS)
scoring = ['accuracy', 'f1_macro', 'precision', 'recall']

scores_cross = cross_validate(
    rf_otimizado,
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


# 8. TREINAR MODELO FINAL
#diabetes_rf = rf_otimizado.fit(atributos_b, classes_b)


# 9. SALVAR MODELO
#dump(diabetes_rf, open('modelo_random_forest.pkl', 'wb'))