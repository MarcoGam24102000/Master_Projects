# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 09:34:38 2021

@author: Samuel Heleno
"""

import numpy as np
import pandas as pd
import skfuzzy as fz
# from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot, pylab


### Análise do relatório do Dataset -> Jupiter
# Dica: apagar a 1ª linha do Excel

# import pandas as pd 
# import pandas_profiling
# A=pd.read_excel('Original_data_group_H.xlsx', sheet_name='Clustering', usecols="A:D") 
# A.profile_report() 



### Importar dataset original
data_set_original = pd.read_excel('Original_data_group_H.xlsx', sheet_name='Clustering', usecols="A:D")            
matriz_data_set_original = np.array(data_set_original)         # Converter dados importados numa matriz
matriz_data_set_original = matriz_data_set_original[:,1:]      # Remover coluna "Consumer ID"



### Tratamento de dados
# Corrigir dados com um método personalizado
matriz_data_set_corrigida = matriz_data_set_original

for i in range(0, len(matriz_data_set_original)):          # Cada linha
    for j in range(0, len(matriz_data_set_original[0])):   # Cada coluna
        if pd.isnull(matriz_data_set_corrigida[i,j]):      # Se o elemento for nulo
            # Proceder à correção de dados, de acordo com cada "feature"
            if j==0:
                matriz_data_set_corrigida[i,j] = int(matriz_data_set_original[i,2] / matriz_data_set_original[i,1])  # Garantir que é inteiro
            elif j==1:
                matriz_data_set_corrigida[i,j] = matriz_data_set_original[i,2] / matriz_data_set_original[i,0]
            else:
                matriz_data_set_corrigida[i,j] = matriz_data_set_original[i,0] * matriz_data_set_original[i,1]
# Corrigir dados com um método genérico alternativo - Média da coluna
# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# imp.fit(matriz_data_set_original)
# matriz_data_set_corrigida = imp.transform(matriz_data_set_original)


# Coeficinete de Correlação entre "features"
cor_matriz_data_set_corrigida = np.corrcoef(matriz_data_set_corrigida, rowvar=False)


# Normalização STANDARD dos dados
scaler = StandardScaler()
scaler.fit(matriz_data_set_corrigida)
print(scaler.fit(matriz_data_set_corrigida))
matriz_data_set_corrigida_normalizada = scaler.transform(matriz_data_set_corrigida)
# Representação gráfica das série originais de dados e das séries já normalizadas
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
ax1.plot(matriz_data_set_corrigida[:,0], label='Tenure')
ax1.plot(matriz_data_set_corrigida[:,1], label='MonthlyCharges')
ax1.plot(matriz_data_set_corrigida[:,2], label='TotalCharges')
ax2.plot(matriz_data_set_corrigida_normalizada[:,0], label='Tenure')
ax2.plot(matriz_data_set_corrigida_normalizada[:,1], label='MonthlyCharges')
ax2.plot(matriz_data_set_corrigida_normalizada[:,2], label='TotalCharges')
# ax1.legend([matriz_data_set_corrigida[:,0], matriz_data_set_corrigida[:,1], matriz_data_set_corrigida[:,2]], ['label1', 'label2', 'label3'])
# ax2.plot(matriz_data_set_corrigida_normalizada)
ax1.legend(framealpha=1, frameon=True);
ax2.legend(framealpha=1, frameon=True);


### Divisão de subsets de treino e de teste
# Subset de treino - Training data
treino_lenght = int(0.7 * len(matriz_data_set_corrigida_normalizada))    
matriz_training_data = matriz_data_set_corrigida_normalizada[0:treino_lenght,:] # Corresponde a 70% das amostras do dataset
# Subset de teste - Test data
matriz_test_data = matriz_data_set_corrigida_normalizada[treino_lenght:,:]      # Corresponde a 30% das amostras do dataset




#######################################
### Aprendizagem não-supervisionada ###
#######################################

### Clustering Hierárquico - abordagem aglomerativa

# Representação gráfica dos objetos do subset de treino (training data)
labels = range(1,treino_lenght)                    # Nº máximo de clusters = nº total de objetos/amostras
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(matriz_training_data[:,0], matriz_training_data[:,1], matriz_training_data[:,2])
for label, x, y, z in zip(labels, matriz_training_data[:, 0], matriz_training_data[:, 1], matriz_training_data[:, 2]):
    ax.text(x, y, z, label)
ax.set_title("Representação gráfica dos objetos do subset de treino")
ax.set_xlabel("Tenure")
ax.set_ylabel("Monthly Charges")
ax.set_zlabel("Total Charges")
plt.grid()


# Cálculo da distância entre objetos
Y_euclidean = pdist(matriz_training_data, metric='euclidean')
Y_euclidean_square = squareform(Y_euclidean)     # Matriz quadrada simétrica com as distâncias entre os objetos
Y_cityblock = pdist(matriz_training_data, metric='cityblock')
Y_cityblock_square = squareform(Y_cityblock)     # Matriz quadrada simétrica com as distâncias entre os objetos


# Formação de grupos com base nas distâncias identificadas entre objetos e/ou entre objetos e clusters (ou mesmo entre clusters)
# Z = linkage(matriz_training_data, 'average') # Distância "averag
Z_euclidean_average = linkage(matriz_training_data, method='average', metric='euclidean') # Distância "average"
Z_euclidean_ward = linkage(matriz_training_data, method='ward', metric='euclidean') # Distância "ward"
Z_cityblock_average = linkage(matriz_training_data, method='average', metric='cityblock') # Distância "average"
# Z_cityblock_ward = linkage(matriz_training_data, method='ward', metric='cityblock')  #Method 'ward' requires the distance metric to be Euclidean


# Representação dos DENDROGRAMAs
labelList = range(1, treino_lenght+1)

plt.figure(figsize=(12, 8)) 
dendrogram(Z_euclidean_average, orientation='top', labels=labelList, distance_sort='descending', show_leaf_counts=True)
plt.show()

plt.figure(figsize=(12, 8)) 
dendrogram(Z_euclidean_ward, orientation='top', labels=labelList, distance_sort='descending', show_leaf_counts=True)

plt.figure(figsize=(12, 8)) 
dendrogram(Z_cityblock_average, orientation='top', labels=labelList, distance_sort='descending', show_leaf_counts=True)
plt.show()

# plt.figure(figsize=(12, 8)) 
# dendrogram(Z_cityblock_ward, orientation='top', labels=labelList, distance_sort='descending', show_leaf_counts=True)
# plt.show()



# Análise do alocação dos objetos quando é definido um nº total de clusters
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,8))
fig.suptitle("Número de objetos por cluster")

C1 = fcluster(Z_euclidean_average, 4, 'maxclust')  # 4 clusters - exemplo
centroid1 = matriz_training_data[C1==1, :].mean(axis=0)
centroid2 = matriz_training_data[C1==2, :].mean(axis=0)
centroid3 = matriz_training_data[C1==3, :].mean(axis=0)
centroid4 = matriz_training_data[C1==4, :].mean(axis=0)
centroids1 = np.vstack((centroid1, centroid2, centroid3, centroid4))
print("Centroides dos 4 clusters Z_euclidean_average=\n", centroids1)


C2 = fcluster(Z_euclidean_ward, 4, 'maxclust')  # 3 clusters - exemplo
centroid1 = matriz_training_data[C2==1, :].mean(axis=0)
centroid2 = matriz_training_data[C2==2, :].mean(axis=0)
centroid3 = matriz_training_data[C2==3, :].mean(axis=0)
centroid4 = matriz_training_data[C2==4, :].mean(axis=0)
centroids2 = np.vstack((centroid1, centroid2, centroid3, centroid4))
print("Centroides dos 4 clusters Z_euclidean_ward=\n", centroids2)


C3 = fcluster(Z_cityblock_average, 4, 'maxclust')  # 3 clusters - exemplo
centroid1 = matriz_training_data[C3==1, :].mean(axis=0)
centroid2 = matriz_training_data[C3==2, :].mean(axis=0)
centroid3 = matriz_training_data[C3==3, :].mean(axis=0)
centroid4 = matriz_training_data[C3==4, :].mean(axis=0)
centroids3 = np.vstack((centroid1, centroid2, centroid3, centroid4))
print("Centroides dos 4 clusters Z_cityblock_average=\n", centroids3)


ax1.hist(C1)
ax1.set_title('Método "average", Métrica "Euclidean"')
ax2.hist(C2)
ax2.set_title('Método "ward", Métrica "Euclidean"')
ax3.hist(C3)
ax3.set_title('Método "average", Métrica "Manhattan/Cityblock"')






# Estudo: Caso novas observações sejam apresentadas, em que clusters é que se vão alojar?
# Concatenar, individualmente, cada novo objeto com os centroides dos clusters considerados
# Calcular as distâncias entre a nova observação e cada centroide
# Matriz qaudrada
# Atribuir a observação ao cluster com menor distância
# Recalcular centroides
# Repetir para o resto das observações/amostras









### Clustering K-mean

# Verificar o efeito de diferentes cenários de inicialização - devido ao facto do método ser eterativo       
# kmeans = KMeans(n_clusters=3, max_iter=5, n_init=5).fit(A)
# Cluster_ID = kmeans.labels_
# centroids = kmeans.cluster_centers_      

# Método do "Cotovelo"
Sum_of_squared_distances = []
n_clusters2 = range(2,20)

for k in n_clusters2 :
    km = KMeans(n_clusters=k, max_iter=300, n_init=5).fit(matriz_training_data)
    Sum_of_squared_distances.append(km.inertia_)


plt.figure(figsize=(10,8))
plt.plot(n_clusters2, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')         

        
# Método Silhouette - análise para um nº variável de clusters
aux = 0
max_silhouette = 0
silhouette_vector = []
n_clusters3 = range(2, treino_lenght) # minimo 2 clusters, máximo (nº objetos-1) clusters
for j in n_clusters3:
    km =KMeans(n_clusters=j, max_iter=300, n_init=5).fit(matriz_training_data)
    labels3 = km.labels_
    silhouette_avg = silhouette_score(matriz_training_data, labels3)
    print("For n_clusters=", j, "The averegae silhouette_score is:", silhouette_avg)
    # Obter o nº de clusters com parâmetro "Silhouette" mais favorável
    aux = silhouette_avg
    if aux > max_silhouette:
        max_silhouette = aux
        number_recommended_clusters = j
    # silhouette_values = silhouette_samples(matriz_training_data, labels3)
    silhouette_vector.append(silhouette_avg)
    # print("For n_clusters = ", j, "silhouette_values are:", silhouette_values)
    
print("\n\nCom base no gráfico Elbow e no método Silhouette, é recomendável formar", number_recommended_clusters, "clusters!")   






# kmeans = KMeans(n_clusters=number_recommended_clusters, max_iter=300, n_init=5).fit(matriz_training_data) # Agrupar as observações em 3 clusters

kmeans = KMeans(n_clusters=4, max_iter=300, n_init=5).fit(matriz_training_data) # Agrupar as observações em 3 clusters
Cluster_ID = kmeans.labels_
centroides_A = kmeans.cluster_centers_  
print("Centroides dos ",number_recommended_clusters, " clusters recomendados:\n", centroides_A)


Cluster_ID_transpose = np.transpose(Cluster_ID).reshape(1,treino_lenght)
objetos_c1 = []
objetos_c2 = []
objetos_c3 = []
objetos_c4 = []


# Filtrar objetos por clusters - Feito para 4 clusters!!!!!!!!
for i in range (0, len(Cluster_ID_transpose[0])):
    if Cluster_ID_transpose[0,i] == 0:
        objetos_c1.append(matriz_training_data[i, :])
    elif Cluster_ID_transpose[0,i] == 1:
        objetos_c2.append(matriz_training_data[i, :]) 
    elif Cluster_ID_transpose[0,i] == 2:
        objetos_c3.append(matriz_training_data[i, :]) 
    elif Cluster_ID_transpose[0,i] == 3:
        objetos_c4.append(matriz_training_data[i, :]) 


list1 = list(zip(*objetos_c1)) 
list2 = list(zip(*objetos_c2)) 
list3 = list(zip(*objetos_c3)) 
list4 = list(zip(*objetos_c4)) 

# # Representação gráfica dos objetos correspondentes a cada cluster, bem como dos centroides
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
# Representar os objetos de cada cluster com cores diferentes
ax.scatter(list(list1[0]),list(list1[1]), list(list1[2]), c="blue")
ax.scatter(list(list2[0]),list(list2[1]), list(list2[2]), c="red")
ax.scatter(list(list3[0]),list(list3[1]), list(list3[2]), c="pink")
ax.scatter(list(list4[0]),list(list4[1]), list(list4[2]), c="green")
# Representar os centroides 
ax.scatter(centroides_A[:,0], centroides_A[:,1], centroides_A[:,2], c="black", s=40)

labels = range(1,treino_lenght)                    # Nº máximo de clusters = nº total de objetos/amostras
labeA = ["Centroide 1", "Centroide 2", "Centroide 3", "Centroide 4"]

for label, x, y, z in zip(labels, matriz_training_data[:, 0], matriz_training_data[:, 1], matriz_training_data[:, 2]):
    ax.text(x, y, z, label)
for label, x, y, z in zip(labeA, centroides_A[:, 0], centroides_A[:, 1], centroides_A[:, 2]):
    ax.text(x, y, z, label)

ax.set_title("Representação gráfica dos objetos correspondentes a cada cluster, bem como dos centroides")
ax.set_xlabel("Tenure")
ax.set_ylabel("Monthly Charges")
ax.set_zlabel("Total Charges")
plt.grid()



##########

# kmeans = KMeans(n_clusters=7, max_iter=300, n_init=5).fit(matriz_training_data) # Agrupar as observações em 3 clusters
# Cluster_ID = kmeans.labels_
# centroides_B = kmeans.cluster_centers_  
# print("Centroides dos ",number_recommended_clusters, " clusters recomendados:\n", centroides_B)


# Cluster_ID_transpose = np.transpose(Cluster_ID).reshape(1,treino_lenght)
# objetos_c1 = []
# objetos_c2 = []
# objetos_c3 = []
# objetos_c4 = []
# objetos_c5 = []
# objetos_c6 = []
# objetos_c7 = []

# # Filtrar objetos por clusters - Feito para 4 clusters!!!!!!!!
# for i in range (0, len(Cluster_ID_transpose[0])):
#     if Cluster_ID_transpose[0,i] == 0:
#         objetos_c1.append(matriz_training_data[i, :])
#     elif Cluster_ID_transpose[0,i] == 1:
#         objetos_c2.append(matriz_training_data[i, :]) 
#     elif Cluster_ID_transpose[0,i] == 2:
#         objetos_c3.append(matriz_training_data[i, :]) 
#     elif Cluster_ID_transpose[0,i] == 3:
#         objetos_c4.append(matriz_training_data[i, :]) 
#     elif Cluster_ID_transpose[0,i] == 4:
#         objetos_c5.append(matriz_training_data[i, :]) 
#     elif Cluster_ID_transpose[0,i] == 5:
#         objetos_c6.append(matriz_training_data[i, :]) 
#     elif Cluster_ID_transpose[0,i] == 6:
#         objetos_c7.append(matriz_training_data[i, :]) 


# list1 = list(zip(*objetos_c1)) 
# list2 = list(zip(*objetos_c2)) 
# list3 = list(zip(*objetos_c3)) 
# list4 = list(zip(*objetos_c4)) 
# list5 = list(zip(*objetos_c5)) 
# list6 = list(zip(*objetos_c6)) 
# list7 = list(zip(*objetos_c7)) 


# # # Representação gráfica dos objetos correspondentes a cada cluster, bem como dos centroides
# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(111, projection='3d')
# # Representar os objetos de cada cluster com cores diferentes
# ax.scatter(list(list1[0]),list(list1[1]), list(list1[2]), c="blue")
# ax.scatter(list(list2[0]),list(list2[1]), list(list2[2]), c="red")
# ax.scatter(list(list3[0]),list(list3[1]), list(list3[2]), c="pink")
# ax.scatter(list(list4[0]),list(list4[1]), list(list4[2]), c="green")
# ax.scatter(list(list5[0]),list(list5[1]), list(list5[2]), c="brown")
# ax.scatter(list(list6[0]),list(list6[1]), list(list6[2]), c="orange")
# ax.scatter(list(list7[0]),list(list7[1]), list(list7[2]), c="yellow")
# # Representar os centroides 
# ax.scatter(centroides_B[:,0], centroides_B[:,1], centroides_B[:,2], c="black", s=40)

# labels = range(1,treino_lenght)                    # Nº máximo de clusters = nº total de objetos/amostras
# labeB = ["Centroide 1", "Centroide 2", "Centroide 3", "Centroide 4", "Centroide 5", "Centroide 6", "Centroide 7"]

# for label, x, y, z in zip(labels, matriz_training_data[:, 0], matriz_training_data[:, 1], matriz_training_data[:, 2]):
#     ax.text(x, y, z, label)
# for label, x, y, z in zip(labeB, centroides_B[:, 0], centroides_B[:, 1], centroides_B[:, 2]):
#     ax.text(x, y, z, label)

# ax.set_title("Representação gráfica dos objetos correspondentes a cada cluster, bem como dos centroides")
# ax.set_xlabel("Tenure")
# ax.set_ylabel("Monthly Charges")
# ax.set_zlabel("Total Charges")
# plt.grid()


### Clustering Fuzzy C-Means  


matrix_training_transpose = np.transpose(matriz_training_data)  ## .reshape(1,treino_lenght)  

fpcs = []
#for k in n_clusters2 :
    
cntr, u, u0, d, ObjFunction, p, fpc = fz.cluster.cmeans(data = matrix_training_transpose, c = 4, m = 2, error = 0.005, maxiter = 300, metric = 'euclidean', init = None, seed = None)
#fpcs.append(fpc)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(cntr[0,0],cntr[0,1], cntr[0,2], c="blue")
ax.scatter(cntr[1,0],cntr[1,1], cntr[1,2], c="red")
ax.scatter(cntr[2,0],cntr[2,1], cntr[2,2], c="pink")
ax.scatter(cntr[3,0],cntr[3,1], cntr[3,2], c="green")

ax.set_title("Representação gráfica dos centroides")
ax.set_xlabel("Tenure")
ax.set_ylabel("Monthly Charges")
ax.set_zlabel("Total Charges")
plt.grid()



cntr, u, u0, d, ObjFunction, p, fpc = fz.cluster.cmeans(data = matrix_training_transpose, c = 7, m = 2, error = 0.005, maxiter = 300, metric = 'euclidean', init = None, seed = None)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(cntr[0,0],cntr[0,1], cntr[0,2], c="blue")
ax.scatter(cntr[1,0],cntr[1,1], cntr[1,2], c="red")
ax.scatter(cntr[2,0],cntr[2,1], cntr[2,2], c="pink")
ax.scatter(cntr[3,0],cntr[3,1], cntr[3,2], c="green")
ax.scatter(cntr[4,0],cntr[4,1], cntr[4,2], c="orange")
ax.scatter(cntr[5,0],cntr[5,1], cntr[5,2], c="yellow")
ax.scatter(cntr[6,0],cntr[6,1], cntr[6,2], c="brown")

ax.set_title("Representação gráfica dos centroides")
ax.set_xlabel("Tenure")
ax.set_ylabel("Monthly Charges")
ax.set_zlabel("Total Charges")
plt.grid()


plt.figure(figsize=(10,8))
plt.plot(n_clusters2, fpcs, 'bx-')
plt.xlabel('k')
plt.ylabel('FPCS')
plt.title('Fuzzy C-Means Method For Optimal k') 
 


 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

