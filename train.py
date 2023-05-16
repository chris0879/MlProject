import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import mlflow

# Caricamento del dataset
data = pd.read_csv('dataset.csv')

# Divisione del dataset in feature e target
X = data.drop('species', axis=1)
y = data['species']

# Divisione del dataset in insiemi di addestramento e di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creazione e addestramento del modello KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Salvataggio del modello addestrato
joblib.dump(knn, 'model.joblib')

# Registrazione del modello in MLflow
with mlflow.start_run():
    mlflow.log_param('n_neighbors', 3)
    mlflow.sklearn.log_model(knn, 'model')
