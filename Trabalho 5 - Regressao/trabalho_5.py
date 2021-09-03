#!/usr/bin/env python
# coding: utf-8

# In[48]:


# Trabalho 5: Regressão Linear:
# José Carlos Silva Gadelha -> matricula: 389110
# André Luís Marques Rodrigues -> matrícula: 374866


# In[49]:


import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # Plotagem


# In[50]:


# Dados

dados = pd.read_csv('kc_house_data.csv', sep=',')
dados.head()
#qnt_dados= len(dados)
#print(dados)


# In[54]:


# X demais atributos e y = preço

X = dados[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']].values
y = dados['price'].values

#print(X)


# In[52]:


# - Utilize a regressão de cumeeira (Ridge Regression), com constante de regularização (λ) igual a 0.01.
# - Utilize validação cruzada com 5 subconjuntos (5-fold cross-validation).
# - Para cada “fold”, indicar o modelo obtido e o seu respectivo coeficiente de terminação ajustado 𝑅𝑎𝑗2.
#  II) O modelo estimado de regressão (linear) “se ajustou” bem aos dados? Justifique.


# In[53]:


for K in range(5):
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, train_size=0.70) # Dividindo os dados em: treinamento (70%) e teste (30%)
    
    identidade = np.eye(X_train.T.shape[0]) # regularização de Tikhonov
    
    #Beta = (np.linalg.inv(X_train.T @ X_train + 0.01* identidade)) @ X_train.T @ y_train  # B = (X.T * X + gama*I)^-1 * X.T * y regularização de Tikhonov
    Beta = (np.linalg.inv(X_train.T @ X_train + 0.01* identidade)) @ X_train.T @ y_train  # B = (X.T * X + gama)^-1 * X.T * y
    
    # Cálculo do y estimado
    y_aprox = X_test @ Beta
    
    
    RSS = ((y_test - y_aprox)**2).sum() # Cálculo da soma dos quadrados do resíduo
      
    TSS = ((y_test - y_test.mean())**2).sum() # Cálculo da soma dos quadrados menos a média
      
    n = X_test.shape[0] # Coeficiente de determinação ajustado
    k = X_test.shape[1]
      
    R2_aj = 1 - (RSS/(n-k))/(TSS/(n-1))
    print(f'Interação {K+1}, valor do coeficiente de determinação ajustado: R2aj = {R2_aj}')


# In[ ]:


# O modelo estimado de regressão linear não se ajustou bem aos dados, pois os valores de ficaram em torno de 0,70, ou seja um valor baixo. 


# In[ ]:


# separa os dados entre treino e teste 
# o interressante dessa função que ela separar os dados de forma aleatoria, não em sequencia.

def separar(dados_a_separar,a,b):
    
    lista = []
    
    qnt_teste = int((len(dados_a_separar))*(a/100))
    qnt_treino = int((len(dados_a_separar))*(b/100))
    
    teste = np.zeros((qnt_teste,21)) # criando um vetor com de n-1 posições
    treino = np.zeros((qnt_treino,21)) # criando um vetor com de n-1 posições
    
    for x in range(0,qnt_teste):
        
        tamanho_lista = len(lista)
        
        while True:
            indice = round(random.uniform(0, 99))
            
            if elemento_lista(lista,indice) == False:
                lista.append(indice)
                break
                
        for y in range(0,6):
               teste[x,y] = dados_a_separar[indice,y]

                        
    for x in range(0,qnt_treino):
        
        tamanho_lista = len(lista)
        
        while True:
            indice = round(random.uniform(0, 99))
            
            if elemento_lista(lista,indice) == False:
                lista.append(indice)
                break
                
        for y in range(0,21):
            treino[x,y] = dados_a_separar[indice,y]
 
    return treino, teste


# In[45]:


def k_fold(dados, k_fold): # esse k 
    
    
    
    aux = 0
    soma = 0
    soma_total = 0
    qnt_dados= len(dados)
    fold = int(qnt_dados / k_fold)
    
    qnt_teste = fold 
    qnt_treino = qnt_dados - fold
    
    teste = np.zeros((qnt_teste,)) # criando um vetor com de n-1 posições
    treino = np.zeros((qnt_treino,6)) # criando um vetor com de n-1 posições
    
    fold01 = np.zeros((fold,21)) # criando um vetor dados teste
    fold02 = np.zeros((fold,21)) # criando um vetor dados teste
    fold03 = np.zeros((fold,21)) # criando um vetor dados teste
    fold04 = np.zeros((fold,21)) # criando um vetor dados teste
    fold05 = np.zeros((fold,21)) # criando um vetor dados teste
    fold06 = np.zeros((fold,21)) # criando um vetor dados teste

    for l in range(0,qnt_dados): # separando os dados nas 5 folds
        for m in range(0,19):
            if l < 3602:
                fold01[l,m] = dados[l,m]
            else:
                if l >= 3602 and l < 7204:
                    fold02[l-3602,m] = dados[l,m]
                else:
                    if l >= 7204 and l < 10806:
                        fold03[l-7204,m] = dados[l,m]
                    else:
                        if l >= 10806 and l < 13868:
                            fold04[l-10806,m] = dados[l,m]
                        else:
                            if l >= 13868 and l < 16930:
                                fold05[l-13868,m] = dados[l,m]
                            else:
                                if l >= 16930:
                                    fold06[l-16930,m] = dados[l,m]
                                    
    for cont in range(0,k_fold):
        
        if( cont == 0):
            
            teste = fold01 # separando uma fold para teste
            treino = fold02 + fold03 + fold04 + fold05 + fold06 # separando 4 folds para treino
            
            treino_test, teste_test = separar(teste,30,70)
            treino_train, teste_train = separar(treino,30,70)
            
          
            X_train = treino_train[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']].values
            y_train = treino_test[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']].values
            X_test = teste_train['price'].values
            y_test = teste_test['price'].values
            
            identidade = np.eye(X_train.T.shape[0]) # regularização de Tikhonov
    
            #Beta = (np.linalg.inv(X_train.T @ X_train + 0.01* identidade)) @ X_train.T @ y_train  # B = (X.T * X + gama*I)^-1 * X.T * y regularização de Tikhonov
            Beta = (np.linalg.inv(X_train.T @ X_train + 0.01* identidade)) @ X_train.T @ y_train  # B = (X.T * X + gama)^-1 * X.T * y

            # Cálculo do y estimado
            y_aprox = X_test @ Beta


            RSS = ((y_test - y_aprox)**2).sum() # Cálculo da soma dos quadrados do resíduo

            TSS = ((y_test - y_test.mean())**2).sum() # Cálculo da soma dos quadrados menos a média

            n = X_test.shape[0] # Coeficiente de determinação ajustado
            k = X_test.shape[1]

            R2_aj = 1 - (RSS/(n-k))/(TSS/(n-1))
            print(f'Interação k = 1, valor do coeficiente de determinação ajustado: R2aj = {R2_aj}')
            
            
        else:
            if( cont == 1):
                
                teste = fold02 # separando uma fold para teste    
                treino = fold01 + fold03 + fold04 + fold05 + fold06
                
                treino_test, teste_test = separar(teste,30,70)
                treino_train, teste_train = separar(treino,30,70)


                X_train = treino_train[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']].values
                y_train = treino_test[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']].values
                X_test = teste_train['price'].values
                y_test = teste_test['price'].values

                identidade = np.eye(X_train.T.shape[0]) # regularização de Tikhonov

                #Beta = (np.linalg.inv(X_train.T @ X_train + 0.01* identidade)) @ X_train.T @ y_train  # B = (X.T * X + gama*I)^-1 * X.T * y regularização de Tikhonov
                Beta = (np.linalg.inv(X_train.T @ X_train + 0.01* identidade)) @ X_train.T @ y_train  # B = (X.T * X + gama)^-1 * X.T * y

                # Cálculo do y estimado
                y_aprox = X_test @ Beta


                RSS = ((y_test - y_aprox)**2).sum() # Cálculo da soma dos quadrados do resíduo

                TSS = ((y_test - y_test.mean())**2).sum() # Cálculo da soma dos quadrados menos a média

                n = X_test.shape[0] # Coeficiente de determinação ajustado
                k = X_test.shape[1]

                R2_aj = 1 - (RSS/(n-k))/(TSS/(n-1))
                print(f'Interação k = 2, valor do coeficiente de determinação ajustado: R2aj = {R2_aj}')
            
            else:
                if( cont == 2):
                    
                    teste = fold03 # separando uma fold para teste
                    treino = fold01 + fold02 + fold04 + fold05 + fold06
                    
                    treino_test, teste_test = separar(teste,30,70)
                    treino_train, teste_train = separar(treino,30,70)


                    X_train = treino_train[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']].values
                    y_train = treino_test[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']].values
                    X_test = teste_train['price'].values
                    y_test = teste_test['price'].values

                    identidade = np.eye(X_train.T.shape[0]) # regularização de Tikhonov

                    #Beta = (np.linalg.inv(X_train.T @ X_train + 0.01* identidade)) @ X_train.T @ y_train  # B = (X.T * X + gama*I)^-1 * X.T * y regularização de Tikhonov
                    Beta = (np.linalg.inv(X_train.T @ X_train + 0.01* identidade)) @ X_train.T @ y_train  # B = (X.T * X + gama)^-1 * X.T * y

                    # Cálculo do y estimado
                    y_aprox = X_test @ Beta


                    RSS = ((y_test - y_aprox)**2).sum() # Cálculo da soma dos quadrados do resíduo

                    TSS = ((y_test - y_test.mean())**2).sum() # Cálculo da soma dos quadrados menos a média

                    n = X_test.shape[0] # Coeficiente de determinação ajustado
                    k = X_test.shape[1]

                    R2_aj = 1 - (RSS/(n-k))/(TSS/(n-1))
                    print(f'Interação k = 3, valor do coeficiente de determinação ajustado: R2aj = {R2_aj}')
                
                else:
                    if( cont == 3):
                        
                        teste = fold04 # separando uma fold para teste 
                        treino = fold01 + fold02 + fold03 + fold05 + fold06
                        
                        treino_test, teste_test = separar(teste,30,70)
                        treino_train, teste_train = separar(treino,30,70)


                        X_train = treino_train[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']].values
                        y_train = treino_test[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']].values
                        X_test = teste_train['price'].values
                        y_test = teste_test['price'].values

                        identidade = np.eye(X_train.T.shape[0]) # regularização de Tikhonov

                        #Beta = (np.linalg.inv(X_train.T @ X_train + 0.01* identidade)) @ X_train.T @ y_train  # B = (X.T * X + gama*I)^-1 * X.T * y regularização de Tikhonov
                        Beta = (np.linalg.inv(X_train.T @ X_train + 0.01* identidade)) @ X_train.T @ y_train  # B = (X.T * X + gama)^-1 * X.T * y

                        # Cálculo do y estimado
                        y_aprox = X_test @ Beta


                        RSS = ((y_test - y_aprox)**2).sum() # Cálculo da soma dos quadrados do resíduo

                        TSS = ((y_test - y_test.mean())**2).sum() # Cálculo da soma dos quadrados menos a média

                        n = X_test.shape[0] # Coeficiente de determinação ajustado
                        k = X_test.shape[1]

                        R2_aj = 1 - (RSS/(n-k))/(TSS/(n-1))
                        print(f'Interação k = 4, valor do coeficiente de determinação ajustado: R2aj = {R2_aj}')
                        
                        
                    else:
                        if( cont == 4):

                            teste = fold05 # separando uma fold para teste
                            treino = fold01 + fold02 + fold03 + fold04 + fold06
                            
                            treino_test, teste_test = separar(teste,30,70)
                            treino_train, teste_train = separar(treino,30,70)


                            X_train = treino_train[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']].values
                            y_train = treino_test[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']].values
                            X_test = teste_train['price'].values
                            y_test = teste_test['price'].values

                            identidade = np.eye(X_train.T.shape[0]) # regularização de Tikhonov

                            #Beta = (np.linalg.inv(X_train.T @ X_train + 0.01* identidade)) @ X_train.T @ y_train  # B = (X.T * X + gama*I)^-1 * X.T * y regularização de Tikhonov
                            Beta = (np.linalg.inv(X_train.T @ X_train + 0.01* identidade)) @ X_train.T @ y_train  # B = (X.T * X + gama)^-1 * X.T * y

                            # Cálculo do y estimado
                            y_aprox = X_test @ Beta


                            RSS = ((y_test - y_aprox)**2).sum() # Cálculo da soma dos quadrados do resíduo

                            TSS = ((y_test - y_test.mean())**2).sum() # Cálculo da soma dos quadrados menos a média

                            n = X_test.shape[0] # Coeficiente de determinação ajustado
                            k = X_test.shape[1]

                            R2_aj = 1 - (RSS/(n-k))/(TSS/(n-1))
                            print(f'Interação k = 5, valor do coeficiente de determinação ajustado: R2aj = {R2_aj}')
                            
                        else:
                                if( cont == 5):

                                    teste = fold05 # separando uma fold para teste
                                    treino = fold01 + fold02 + fold03 + fold04 + fold06

                                    treino_test, teste_test = separar(teste,30,70)
                                    treino_train, teste_train = separar(treino,30,70)


                                    X_train = treino_train[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']].values
                                    y_train = treino_test[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']].values
                                    X_test = teste_train['price'].values
                                    y_test = teste_test['price'].values

                                    identidade = np.eye(X_train.T.shape[0]) # regularização de Tikhonov

                                    #Beta = (np.linalg.inv(X_train.T @ X_train + 0.01* identidade)) @ X_train.T @ y_train  # B = (X.T * X + gama*I)^-1 * X.T * y regularização de Tikhonov
                                    Beta = (np.linalg.inv(X_train.T @ X_train + 0.01* identidade)) @ X_train.T @ y_train  # B = (X.T * X + gama)^-1 * X.T * y

                                    # Cálculo do y estimado
                                    y_aprox = X_test @ Beta


                                    RSS = ((y_test - y_aprox)**2).sum() # Cálculo da soma dos quadrados do resíduo

                                    TSS = ((y_test - y_test.mean())**2).sum() # Cálculo da soma dos quadrados menos a média

                                    n = X_test.shape[0] # Coeficiente de determinação ajustado
                                    k = X_test.shape[1]

                                    R2_aj = 1 - (RSS/(n-k))/(TSS/(n-1))
                                    print(f'Interação k = 6, valor do coeficiente de determinação ajustado: R2aj = {R2_aj}')    
                    
                                
    


# In[ ]:


# k_fold(dados, 6)


# In[47]:


# tentamos implementar o k fold, nesse caso para k = 6 pois a divisão é exata,
# mais esta dando um erro. devido esse trabalho esta sendo enviado em atraso
# Melhor enviar assim mesmo ( mais temos o calculo de Rj_Aj para 5 interações) .... 
# Gostaria que se possivel analisasse a logica e retornasse para termos o feedback 
# Que estavamos no caminho certo.


# In[38]:


#2.2) Regressão Polinomial

# Fizemos esse item no matlab, pois apresentou-se ser mais pratico desenvolver esse problema nele.

