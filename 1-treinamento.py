from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd
import pandas as pd
import unidecode
from sklearn.preprocessing import LabelEncoder
import joblib

# Funções de normalização e mapeamento de bancos de dados
def normalize_text(v):
    if v is None:
        return "desconhecido"
    v = str(v)
    v = unidecode.unidecode(v).lower().strip().replace(" ", "")
    return v if v != "" else "desconhecido"

bancos_sql = {'sqlserver', 'mysql', 'postgresql', 'oracle', 'googlebigquery', 'sqlite', 'saphana', 'snowflake', 'amazonauroraourds', 'mariadb', 'db2', 'firebird', 'amazonredshift', 'microsoftaccess'}
bancos_nosql = {'s3', 'databricks', 'amazonathena', 'mongodb', 'hive', 'dynamodb', 'presto', 'elaticsearch', 'redis', 'firebase', 'splunk', 'nenhum', 'cassandra', 'hbase', 'googlefirestore', 'neo4j', 'excel'}

def map_bancos(col):
    col = col.fillna("").astype(str).str.lower()
    sql_count = col.apply(lambda x: sum(1 for s in bancos_sql if s in x))
    nosql_count = col.apply(lambda x: sum(1 for s in bancos_nosql if s in x))
    return pd.DataFrame({"sql_count": sql_count, "nosql_count": nosql_count})


# FUNÇÃO PRINCIPAL DE PRÉ-PROCESSAMENTO
def preprocess_data(df, le_dict=None, is_training=True):
    # 1. Normalizar strings
    df = df.applymap(lambda x: normalize_text(x) if isinstance(x, str) else x)

    # 2. Mapear bancos SQL/NoSQL
    bancos_mapped = map_bancos(df["bancos_de_dados"])
    df = pd.concat([df.reset_index(drop=True), bancos_mapped.reset_index(drop=True)], axis=1)
    df = df.drop(columns=["bancos_de_dados"])

    # 3. Tratar idade
    df['idade'] = pd.to_numeric(df['idade'], errors='coerce').fillna(0).astype(int)

    # 4. Remover colunas do modelo
    df = df.drop(["vive_no_brasil", "pcd"], axis=1)

    # 5. Mapear cargos (apenas no treinamento)
    if is_training:
        mapeamento_cargos = {
            'analistadebi/bianalyst': 'analiseeestrategia',
            'analistadenegocios/businessanalyst': 'analiseeestrategia',
            'analistadeinteligenciademercado/marketintelligence': 'analiseeestrategia',
            'analistademarketing': 'analiseeestrategia',
            'economista': 'analiseeestrategia',
            'productmanager/productowner(pm/apm/dpm/gpm/po)': 'analiseeestrategia',
            'cientistadedados/datascientist': 'cienciadedadoseestatistica',
            'engenheirodemachinelearning/mlengineer': 'cienciadedadoseestatistica',
            'estatistico': 'cienciadedadoseestatistica',
            'analistadedados/dataanalyst': 'cienciadedadoseestatistica',
            'engenheirodedados/arquitetodedados/dataengineer/dataarchitect': 'engenhariadedados',
            'analyticsengineer': 'engenhariadedados',
            'dba/administradordebancodedados': 'engenhariadedados',
            'desenvolvedor/engenheirodesoftware/analistadesistemas': 'engenhariadesoftware',
            'analistadesuporte/analistatecnico': 'engenhariadesoftware',
            'outrasengenharias(naoincluidev)': 'engenhariadesoftware',
            'professor': 'educacaoesuportetecnico',
            'outraopcao': 'educacaoesuportetecnico',
        }
        df["cargo"] = df["cargo"].map(lambda x: mapeamento_cargos.get(x, "outraopcao"))
        df = df.dropna(subset=["cargo"])


    # 6. Codificar colunas categóricas
    categorical_cols = ["genero", "etnia", "estado_moradia", "nivel_ensino", "formacao",
                        "tempo_experiencia_dados", "linguagens_preferidas", "cloud_preferida"]
    
    if is_training:
        le_dict = {}
        for col in categorical_cols + ["cargo"]:
            le = LabelEncoder()
            df[col] = df[col].fillna("desconhecido")
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
        return df, le_dict
    else:
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("desconhecido")
                try:
                    df[col] = le_dict[col].transform(df[col])
                except ValueError:
                    df[col] = 0
        return df

# Carregar dados
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

# Treinar modelo
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Avaliar
y_pred = rf_model.predict(X_test)
print(f"\n✅ Acurácia: {accuracy_score(y_test, y_pred):.4f}")

# Salvar modelo + encoders + colunas
joblib.dump({
    "model": rf_model,
    "le_dict": le_dict,
    "feature_columns": X.columns.tolist()
}, "modelo_cargos.pkl")

print("🚀 Modelo salvo em 'modelo_cargos.pkl'")