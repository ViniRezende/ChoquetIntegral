#Função para o cálculo da Integral de Choquet.

#Requisitos: Fornecer uma matriz de decisão (já normalizada);
            #Fornecer as Capacidades de Choquet (Medida Fuzzy) em ordem lexicográfica.

#OBS: Caso queira simular uma matriz de decisão de uma alternativa, certificar de que a matriz P seja do tipo:
  # P = [[a1, a2, ..., an]]
  #Caso a matriz P seja fornecida da seguinte forma:
    # P = [a1, a2, ...., an], len(P[0]) não sera executado corretamente.

#Bibliotecas utilizadas
import numpy as np
import itertools

def choquetintegral(P, u):
  
  #Etapa 1) Criando uma matriz com todas as coligações em ordem lexicográfica: 
  n = len(P[0]) #Número de critérios
  p = np.arange(1,n+1) #Vetor 1,2...n para ser utilizado como suporte para obter todas as combinações em ordem lexicográfica
  S = np.concatenate([np.pad(np.array(list(itertools.combinations(p, r))), ((0,0),(0,n-r)), 'constant', constant_values=0) for r in range(0,n+1)], axis=0)
    #^ Matriz que contem todas as combinações em ordem lexicográfica   

  #Etapa 2) Processo de ordenamento das alternativas em ordem crescente e coletando e ordenando seus índices 
  P_aumentado = np.pad(P, ((0,0),(0,1)), 'constant', constant_values=0) #Aumenta a matriz de decisão e acrescenta uma coluna de zeros
  P_new = np.sort(P_aumentado) #Ordenamento da matriz aumentada
  P_indice = np.argsort(P) + 1  #Mantendo os índices da ordenação acrescidos de 1 para dar match com a matris S

  M = np.zeros((len(P), 2**n)) #Matriz de zeros para ser substituida pelos valores de  P_new

  #Etapa 3) Cálculo da Integral de Choquet
  for i in range(0,len(P)): #Roda para todas as linhas (alternativas)
    for j in range(1, n+1): #Interaçoes que acontecem nas colunas
      P_indice_aux = np.pad(np.sort(P_indice[i][j-1:n]), ((0,j-1)), 'constant', constant_values=0)  #Obtendo p/ cada interação os índices ordenados acrescidos de zeros
      M[i][np.flatnonzero(np.all(S== P_indice_aux, 1))] = P_new[i][j] - P_new[i][j-1] #Substituindo onde da Match entre S e P_indice_aux na matrix M os valores de P_new

  M  #Matriz de interesse
  CI = np.dot(M, u)
  return CI
