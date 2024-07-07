# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 21:12:39 2021

@author: Samuel Heleno
"""


import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


### Análise do relatório do Dataset -> Jupiter
# Dica: apagar a 1ª linha do Excel

# import pandas as pd 
# import pandas_profiling
# A=pd.read_excel('Original_data_group_H.xlsx', sheet_name='Classification', usecols="A:J") 
# A.profile_report() 



### Importar dataset original
data_set_original = pd.read_excel('Original_data_group_H.xlsx', sheet_name='Classification', usecols="A:J")        
matriz_data_set_original = np.array(data_set_original)         # Converter dados importados numa matriz
matriz_data_set_original = matriz_data_set_original[:,1:]      # Remover coluna "Consumer ID"







### Tratamento de dados

##  Remover amostras que não-representativas ou de fraca qualidade
matriz_data_set_corrigida = matriz_data_set_original
for i in range(0, len(matriz_data_set_corrigida)):          # Cada linha
    for j in range(0, len(matriz_data_set_corrigida[0])):   # Cada coluna
        if pd.isnull(matriz_data_set_corrigida[i,j]) or matriz_data_set_corrigida[i,j]== ' ' :      # Se o elemento for nulo ou estiver em "branco"
            indice = i # indice da amostra         
matriz_data_set_corrigida = np.concatenate( (matriz_data_set_corrigida[0:indice, :], matriz_data_set_corrigida[indice+1 :, :]  ), axis = 0)




## Análise da correlação entre features 
ColumnNames=['Dependents', 'Tenure', 'InternetService', 'StreamingMovies', 'Contract', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churns']
matriz_dataset_ = pd.DataFrame(data=matriz_data_set_corrigida, columns=ColumnNames)


# Análise numérica-categórica -> Método ANOVA
CategoryGroupLists1 = matriz_dataset_.groupby('Churns')['Tenure'].apply(list)
CategoryGroupLists2 = matriz_dataset_.groupby('Churns')['MonthlyCharges'].apply(list)
CategoryGroupLists3 = matriz_dataset_.groupby('Churns')['TotalCharges'].apply(list)


sns.catplot(x="Tenure", y="Churns", kind="box", data=matriz_dataset_)       # Forma visual - Boxplot
sns.catplot(x="MonthlyCharges", y="Churns", kind="box", data=matriz_dataset_)
sns.catplot(x="TotalCharges", y="Churns", kind="box", data=matriz_dataset_)

AnovaResults1 = f_oneway(*CategoryGroupLists1)   # Performing the ANOVA test
AnovaResults2 = f_oneway(*CategoryGroupLists2)   # Performing the ANOVA test
AnovaResults3 = f_oneway(*CategoryGroupLists3)   # Performing the ANOVA test

print('P-Value for Anova (Churns - Tenure) is: ', AnovaResults1[1]) # We accept the Assumption(H0) only when P-Value &gt; 0.05
print('P-Value for Anova (Churns - MonthlyCharges) is: ', AnovaResults2[1]) # We accept the Assumption(H0) only when P-Value &gt; 0.05
print('P-Value for Anova (Churns - TotalCharges) is: ', AnovaResults3[1]) # We accept the Assumption(H0) only when P-Value &gt; 0.05


# Análise categórica-categórica -> Método Chi-Square Test

# Cross tabulation between GENDER and APPROVE_LOAN
CrosstabResult1 = pd.crosstab(index=matriz_dataset_['Dependents'],columns=matriz_dataset_['Churns'])
CrosstabResult2 = pd.crosstab(index=matriz_dataset_['InternetService'],columns=matriz_dataset_['Churns'])
CrosstabResult3 = pd.crosstab(index=matriz_dataset_['StreamingMovies'],columns=matriz_dataset_['Churns'])
CrosstabResult4 = pd.crosstab(index=matriz_dataset_['Contract'],columns=matriz_dataset_['Churns'])
CrosstabResult5 = pd.crosstab(index=matriz_dataset_['PaymentMethod'],columns=matriz_dataset_['Churns'])

print(CrosstabResult1)
print(CrosstabResult2)
print(CrosstabResult3)
print(CrosstabResult4)
print(CrosstabResult5)

# Performing Chi-sq test
ChiSqResult1 = chi2_contingency(CrosstabResult1)
ChiSqResult2 = chi2_contingency(CrosstabResult2)
ChiSqResult3 = chi2_contingency(CrosstabResult3)
ChiSqResult4 = chi2_contingency(CrosstabResult4)
ChiSqResult5 = chi2_contingency(CrosstabResult5)
 
# P-Value is the Probability of H0 being True
# If P-Value&gt;0.05 then only we Accept the assumption(H0)
 
print('The P-Value of the ChiSq Test of (Churns - Dependents) is:', ChiSqResult1[1])
print('The P-Value of the ChiSq Test of (Churns - InternetService) is:', ChiSqResult2[1])
print('The P-Value of the ChiSq Test of (Churns - StreamingMovies) is:', ChiSqResult3[1])
print('The P-Value of the ChiSq Test of (Churns - Contract) is:', ChiSqResult4[1])
print('The P-Value of the ChiSq Test of (Churns - DependenPaymentMethodts) is:', ChiSqResult5[1])





## Normalização STANDARD dos dados
# Dica: dividir a matri em duas (uma com variáveis numéricas e outra com categoricas)
# Normalizar a matriz numérica
# Concatenar as duas
# Inicialmente não se efetua




## Dividir as amostras do dataset em subsets
train_data, test_data, labels_train_data, labels_test_data = train_test_split(matriz_data_set_corrigida[:, 0:8], 
                                                                              matriz_data_set_corrigida[:, 8], 
                                                                              test_size = 0.2, 
                                                                              random_state = 42, shuffle = True, 
                                                                              stratify = matriz_data_set_corrigida[:, 8])
# # Subset de treino - Train data
# train_data_aux = int(0.64 * len(matriz_data_set_corrigida))    
# train_data = matriz_data_set_corrigida[0:train_data_aux,0:8] # Corresponde a 64% das amostras do dataset
# labels_train_data = matriz_data_set_corrigida[0:train_data_aux, 8] # Labels (Output correto)
# # Subset de validação - Validation data
# validation_data_aux = int(0.8 * len(matriz_data_set_corrigida))    
# validation_data = matriz_data_set_corrigida[train_data_aux:validation_data_aux, 0:8] # Corresponde a 16% das amostras do dataset
# labels_validation_data = matriz_data_set_corrigida[train_data_aux:validation_data_aux, 8] # Labels
# # Subset de teste - Test data
# test_data = matriz_data_set_corrigida[validation_data_aux:, 0:8]      # Corresponde a 20% das amostras do dataset
# labels_test_data = matriz_data_set_corrigida[validation_data_aux:, 8] 







###################################
### Aprendizagem supervisionada ###
###################################

### Classificação KNN
# não apropriado para ser aplicado em problemas para os quais as features são categóricas (as distâncias entre dimensões devem ser numéricas).

## Converter variáveis categóricas em núemricas
data_set_knn_adaptado = np.zeros((len(matriz_data_set_corrigida),len(matriz_data_set_corrigida[0])))

def data_adapt(data) :
    # data_adaptado = np.zeros((len(data),len(data[0])))
    data_adaptado = np.copy(data)

    for i in range(0, len(data)):
        # Dependents
        if data_adaptado[i,0] == 'Yes':
            data_adaptado[i,0] = 1
        elif data_adaptado[i,0] == 'No':
            data_adaptado[i,0] = 0
        # InternetServiecs
        if data_adaptado[i,2] == 'No':
            data_adaptado[i,2] = 0
        elif data_adaptado[i,2] == 'Fiber optic':
            data_adaptado[i,2] = 1
        elif data_adaptado[i,2] == 'DSL':
            data_adaptado[i,2] = 2         
        # Streaming Movies
        if data_adaptado[i,3] == 'No internet service':
            data_adaptado[i,3] = 0
        elif data_adaptado[i,3] == 'No':
            data_adaptado[i,3] = 1
        elif data_adaptado[i,3] == 'Yes':
            data_adaptado[i,3] = 2           
        # Contract
        if data_adaptado[i,4] == 'Month-to-month':
            data_adaptado[i,4] = 0
        elif data_adaptado[i,4] == 'One year':
            data_adaptado[i,4] = 1
        elif data_adaptado[i,4] == 'Two year':
            data_adaptado[i,4] = 2
        # Payment Method
        if data_adaptado[i,5] == 'Electronic check':
            data_adaptado[i,5] = 0
        elif data_adaptado[i,5] == 'Bank transfer (automatic)':
            data_adaptado[i,5] = 1
        elif data_adaptado[i,5] == 'Mailed check':
            data_adaptado[i,5] = 2
        elif data_adaptado[i,5] == 'Credit card (automatic)':
            data_adaptado[i,5] = 3
        # # Churns
        # if matriz_data_set_corrigida[i,8] == 'Yes':
        #     data_set_normalized[i,8] = 1
        # elif matriz_data_set_corrigida[i,8] == 'No':
        #     data_set_normalized[i,8] = 0    
    return data_adaptado

train_data_adaptado = data_adapt(train_data) # Adaptação


# Normalização standard
# Normalização STANDARD dos dados
scaler = StandardScaler()
scaler.fit(train_data_adaptado)
print(scaler.fit(train_data_adaptado))
train_data_adaptado_normalizada = scaler.transform(train_data_adaptado)
# Representação gráfica das série originais de dados e das séries já normalizadas
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
ax1.plot(train_data_adaptado[:,0], label='Dependents')
ax1.plot(train_data_adaptado[:,1], label='Tenure')
ax1.plot(train_data_adaptado[:,2], label='InternetService')
ax1.plot(train_data_adaptado[:,3], label='StreamingMovies')
ax1.plot(train_data_adaptado[:,4], label='Contract')
ax1.plot(train_data_adaptado[:,5], label='PaymentMethod')
ax1.plot(train_data_adaptado[:,6], label='MonthlyCharges')
ax1.plot(train_data_adaptado[:,7], label='TotalCharges')
ax2.plot(train_data_adaptado_normalizada[:,0], label='Dependents')
ax2.plot(train_data_adaptado_normalizada[:,1], label='Tenure')
ax2.plot(train_data_adaptado_normalizada[:,2], label='InternetService')
ax2.plot(train_data_adaptado_normalizada[:,3], label='StreamingMovies')
ax2.plot(train_data_adaptado_normalizada[:,4], label='Contract')
ax2.plot(train_data_adaptado_normalizada[:,5], label='PaymentMethod')
ax2.plot(train_data_adaptado_normalizada[:,6], label='MonthlyCharges')
ax2.plot(train_data_adaptado_normalizada[:,7], label='TotalCharges')
# ax1.legend([matriz_data_set_corrigida[:,0], matriz_data_set_corrigida[:,1], matriz_data_set_corrigida[:,2]], ['label1', 'label2', 'label3'])
# ax2.plot(matriz_data_set_corrigida_normalizada)
ax1.legend(framealpha=1, frameon=True);
ax2.legend(framealpha=1, frameon=True);


# for a in range (1,36) :
#     knn = KNeighborsClassifier(n_neighbors=a, metric='euclidean')
#     print("\nTeste Cross_val_score")
#     print(np.mean(cross_val_score(knn, train_data_adaptado, labels_train_data, cv=5)))

# Parametrizar o algoritmo
knn = KNeighborsClassifier(n_neighbors=2, metric='euclidean') # comando para permitir a classificação baseada em vizinhos mais próximos . 
#Neste caso, os 3 vizinhos mais próximos serão usados para votar durante o processo de classificação. 
#Por omissão, a distância de Minkowski (generalização de distâncias Euclideana e de Manhattan, baseada no espaço vetor normalizado) é a considerada. 
#Neste caso, a distância Euclideana foi adotada.

# for a in range(1,36):
#     knn = KNeighborsClassifier(n_neighbors=a, metric='euclidean')
#     print(np.mean(cross_val_score(knn, train_data_adaptado, labels_train_data, scoring='accuracy', cv=5)))
    
    
# K-Fold Cross Validation, onde k=5
print("\nTeste Cross_val_score")
print(np.mean(cross_val_score(knn, train_data_adaptado, labels_train_data, cv=5))) 


# Treino
knn.fit (train_data_adaptado_normalizada, labels_train_data) # para calibrar o modelo de acordo com entradas/saída.
# Resultados do treino
labels_predicted_KNN= knn.predict(train_data_adaptado_normalizada) # baseado no modelo criado anteriormente, permite classificar objetos (neste caso, os mesmos usados durante o período de treino).
print("Resultados treino:")
print(confusion_matrix(labels_train_data, labels_predicted_KNN)) # para criar uma matriz de confusão, de modo a que os valores reais e previstos sejam confrontados.
print(accuracy_score (labels_train_data, labels_predicted_KNN)) # para obter o resultado da exatidão do modelo
print(classification_report (labels_train_data, labels_predicted_KNN)) # para obter um relatório com a exatidão da classificação (precisão, recall e F1-score)



# Testar o algoritmo - Resultados do teste 
test_data_adaptado = data_adapt(test_data)
scaler.fit(test_data_adaptado)
test_data_adaptado_normalizada = scaler.transform(test_data_adaptado)
labels_predicted_test_KNN = knn.predict(test_data_adaptado_normalizada)
print("Resultado teste")
print (confusion_matrix(labels_test_data, labels_predicted_test_KNN))
print (accuracy_score(labels_test_data, labels_predicted_test_KNN))
print (classification_report(labels_test_data, labels_predicted_test_KNN))








### Classificação ANN

# Parametrizar o algoritmo
# Dica: Função de ativaçõa -> logistic
Mdl = MLPClassifier (hidden_layer_sizes=2, activation= 'relu', solver='lbfgs', verbose=True, early_stopping=True, validation_fraction=0.16 ) # para definir uma camada
# intermédia com 3 neurónios, função logística como função de ativação na camada oculta, solver ‘lbfgsg’ definido como algoritmo específico de aprendizagem, ativando-se verbose como True
# para permitir a visualização da função objetivo em cada iteração. É também possível habilitar interrupção prematura do treino (técnica de validação cruzada) usando validation_fraction de
# 0.15 – para que 15% dos dados de treino sejam usados como dados de validação para interrupção prematura do processo de treino – não usada neste caso.

# # Definir função de ativação -> softmaw  
# Mdl.out_activation_ = 'softmax'


# Treino
Mdl = Mdl.fit (train_data_adaptado_normalizada, labels_train_data) # para calibrar o modelo de acordo com as entradas/saídas

# Resultados do treino
labels_predicted_ANN = Mdl.predict (train_data_adaptado_normalizada) # para prever etiquetas baseadas nas entradas apresentadas

print("ANN")

print(confusion_matrix(labels_train_data, labels_predicted_ANN)) # para criar uma matriz de confusão, de modo a que os valores reais e previstos sejam confrontados.
print(accuracy_score (labels_train_data, labels_predicted_ANN)) # para obter o resultado da exatidão do modelo
print(classification_report (labels_train_data, labels_predicted_ANN)) # para obter um relatório com a exatidão da classificação (precisão, recall e F1-score)


# Testar o algoritmo - Resultados do teste
labels_predicted_ANN_Test = Mdl.predict(test_data_adaptado_normalizada) # para prever as saídas baseadas nas entradas do conjunto de dados de teste

print (confusion_matrix(labels_test_data, labels_predicted_ANN_Test))
print (accuracy_score(labels_test_data, labels_predicted_ANN_Test))
print (classification_report(labels_test_data, labels_predicted_ANN_Test))


# De modo a avaliar os parâmetros da rede neuronal:
pesos = Mdl.coefs_ # para avaliar os pesos das redes nas diferentes camadas;
offsets = Mdl.intercepts_ # para avaliar os offsets das redes nas diferentes camadas;
print("Pesos: \n", pesos, "\nOffsets: \n", offsets)

# De modo a avaliar a probabilidade de cada ponto pertencer a cada classe específica:
# probs_Train = Mdl.predict_proba (train_data_adaptado_normalizada)
# probs_Test = Mdl.predict_proba (test_data_adaptado_normalizada)
# print("Prob. Train: \n ", probs_Train, "\nProb. Test: \n", probs_Test)








### Classificação SVM

clf = SVC (C=2,kernel='linear')
clf = clf.fit(train_data_adaptado_normalizada, labels_train_data)
training_predicted_SVM = clf.predict(train_data_adaptado_normalizada)

print("SVC")

print(confusion_matrix(labels_train_data, training_predicted_SVM))
print('Accuracy Score:', accuracy_score (labels_train_data, training_predicted_SVM))
print (classification_report (labels_train_data, training_predicted_SVM)) 

labels_predicted_SVM_Test = clf.predict(test_data_adaptado_normalizada)

print (confusion_matrix(labels_test_data, labels_predicted_SVM_Test))
print (accuracy_score(labels_test_data, labels_predicted_SVM_Test))
print (classification_report(labels_test_data, labels_predicted_SVM_Test))


# indexes = clf.support_
# #sv = clf.supportvectors_ 
# n_sv = clf.n_support_

# print("Atributos SVM:")
# print("clf.support_: \n ", indexes, "\nclf.n_support_: \n", n_sv) 

# ## Aplication of Grid Search to Tune SVM Parameters

# find_parameters =  [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# clf= GridSearchCV(SVC(),find_parameters,cv=3)

# clf.fit(train_data_adaptado_normalizada, labels_train_data) 
# clf.best_params_   







