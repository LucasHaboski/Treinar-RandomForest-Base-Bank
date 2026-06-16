import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import warnings
warnings.filterwarnings('ignore')

# ── 1. Carregar os dados ──────────────────────────────────────────────────────
df = pd.read_csv('bank.csv', sep=';')

print(f'Shape: {df.shape}')
print(df.head())
print()
df.info()
print()

print('Valores ausentes por coluna:')
print(df.isnull().sum())
print()

print('Distribuição da variável y:')
print(df['y'].value_counts())
print(f'\nPorcentagem:\n{df["y"].value_counts(normalize=True).round(3) * 100}')
print()

# ── 2. Pré-processamento ──────────────────────────────────────────────────────
categoricas = df.select_dtypes(include='object').columns.tolist()
print('Colunas categóricas:', categoricas)

df_encoded = df.copy()
le = LabelEncoder()

for col in categoricas:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop(columns=['y'])
y = df_encoded['y']

print(f'\nFeatures: {X.shape[1]} colunas | Amostras: {X.shape[0]} linhas')

# ── 3. Divisão treino / teste ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'Treino: {X_train.shape[0]} amostras')
print(f'Teste:  {X_test.shape[0]} amostras')

# ── 4. Treinar o modelo Random Forest ────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
print('\nModelo treinado com sucesso!')

# ── 5. Avaliação ──────────────────────────────────────────────────────────────
y_pred = rf.predict(X_test)

acc_treino = accuracy_score(y_train, rf.predict(X_train))
acc_teste  = accuracy_score(y_test, y_pred)

print(f'\nAcurácia no treino: {acc_treino:.4f} ({acc_treino*100:.2f}%)')
print(f'Acurácia no teste:  {acc_teste:.4f} ({acc_teste*100:.2f}%)')

if acc_treino - acc_teste > 0.05:
    print('Aviso: possível overfitting.')
else:
    print('Modelo generaliza bem (sem overfitting significativo).')

print()
print('Relatório de Classificação:')
print(classification_report(y_test, y_pred, target_names=['Não assinou (0)', 'Assinou (1)']))

# ── 6. Matriz de confusão ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Não (0)', 'Sim (1)']).plot(
    ax=ax, cmap='Blues'
)
ax.set_title('Matriz de Confusão - Random Forest')
plt.tight_layout()
plt.show()

# ── 7. Importância das features ───────────────────────────────────────────────
importancias = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print('Top 5 features mais importantes:')
print(importancias.head())

plt.figure(figsize=(10, 6))
sns.barplot(x=importancias.values, y=importancias.index, palette='viridis')
plt.title('Importância das Features - Random Forest')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
