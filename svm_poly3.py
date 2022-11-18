# Para cargar los datos de entrenamiento y prueba
import pandas as pd
import pickle
import sys

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score,accuracy_score, precision_score, roc_auc_score, precision_score, recall_score

from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

#with open('train_test_data_down.pkl', 'rb') as f: #5%
    #X_train, X_test, y_train, y_test = pickle.load(f)
    
with open('train_test_data_down_1_9.pkl', 'rb') as f: #3%
    X_train, X_test, y_train, y_test = pickle.load(f)   

X_train.drop(columns=['RECORD_ID'], inplace=True)
X_test.drop(columns=['RECORD_ID'], inplace=True)
y_train.drop(columns=['RECORD_ID'], inplace=True)
y_test.drop(columns=['RECORD_ID'], inplace=True)

kernel = 'poly'
degree = 3

for c in [0.1, 0.5, 1]:
    archivo = f'svc_kernel{kernel}_degree{degree}_c{str(c).replace(".","")}'
    modelo = SVC(kernel=kernel, C=c, degree=degree, random_state=42)

    scores = cross_validate(modelo, X_train, y_train["flg_preeclampsia"], cv=5, scoring=["roc_auc", "accuracy", "precision", "recall"], verbose=5, n_jobs=-1)
    resultados = [scores['test_roc_auc'], scores['test_accuracy'], scores['test_precision'], scores['test_recall']]

    with open(f'scores_{archivo}.txt', 'w') as f:
        for resultado in resultados:
            f.write(str(resultado))
            f.write('\n')

    with open(f'resultado_{archivo}.pkl', 'wb') as f:
        pickle.dump(scores, f)
