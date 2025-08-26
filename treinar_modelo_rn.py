# treinar_modelo_rn.py -> Nome do arquivo
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier # Importa a classe para a rede neural
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import pandas as pd
from preprocess import preprocess_data

# Carregar o dataset
url = "https://raw.githubusercontent.com/Weverton-Cristian/Processing-Dataset_-Intelligent_Agent/master/dataset.csv"
df = pd.read_csv(url)

# Processar os dados usando a função única
df_processed, le_dict = preprocess_data(df, is_training=True)

# Separar X e y
X = df_processed.drop("cargo", axis=1)
y = df_processed["cargo"]

# Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# O MLPClassifier é uma rede neural simples, mas eficaz para classificação.
# Os parâmetros definem a arquitetura e o comportamento do treinamento.
mlp_model = MLPClassifier(
    hidden_layer_sizes=(100, 50), # Duas camadas ocultas
    max_iter=500,                  
    random_state=42,
    solver='adam',                 
    verbose=True,                 
    early_stopping=True,          
    n_iter_no_change=20           
)
mlp_model.fit(X_train, y_train)

# Avaliar o modelo
y_pred = mlp_model.predict(X_test)

# Exibindo as métricas
print("\n--- Métricas de Avaliação da Rede Neural ---")
print(f"✅ **Acurácia:** {accuracy_score(y_test, y_pred):.4f}")
print("\n--- Matriz de Confusão ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Relatório de Classificação (Precisão, Recall e F1-Score) ---")
print(classification_report(y_test, y_pred))

# Salvando o modelo e os encoders
le_cargo = le_dict["cargo"]
joblib.dump({
    "model": mlp_model,
    "le_dict": le_dict,
    "feature_columns": X.columns.tolist(),
    "cargo_classes": le_cargo.classes_.tolist()
}, "modelo_rede_neural.pkl")

print("\n🚀 Modelo de Rede Neural salvo em 'modelo_rede_neural.pkl'")