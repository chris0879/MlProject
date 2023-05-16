import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import mlflow

# Caricamento del dataset di test
data = pd.read_csv('dataset.csv')

# Caricamento del modello addestrato
knn = mlflow.sklearn.load_model('model')

# Divisione del dataset di test in feature e target
X_test = data.drop('species', axis=1)
y_test = data['species']

# Previsione sulle misure dei fiori nel set di test
y_pred = knn.predict(X_test)

# Valutazione delle prestazioni del modello
accuracy = accuracy_score(y_test, y_pred)
print("Accuratezza del modello: {:.2f}".format(accuracy))

# Registrazione dell'accuratezza in MLflow
with mlflow.start_run():
    mlflow.log_metric('accuracy', accuracy)
