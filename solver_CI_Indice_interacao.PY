"""

**Solver para identificação das capacidades de choquet**

**Objetivo:** Gerar os dados (matrizes) necessários para ser aplicado em um solver de otimização com o objetivo de obter os valores das capacidades de Choquet/Fuzzy Measure (FM).

**1) Etapa: Obter uma matriz (M) que contem as diferenças ordenadas entre um valor da matriz de decisão (P) e o seu sucessor nas posições exatas.**
"""

#Bibliotecas utilizadas
import numpy as np
import itertools

#1) Etapa

#Requisitos: Fornecer uma matriz de decisão (já normalizada);
#Dados iniciais: i) P: Matriz de decisão (já normalizada)

#OBS: Caso queira simular uma matriz de decisão de uma alternativa, certificar de que a matriz P seja do tipo:
  # P = [[a1, a2, ..., an]]
  #Caso a matriz P seja fornecida da seguinte forma:
    # P = [a1, a2, ...., an], len(P[0]) não sera executado corretamente.

def matrizM(P):
  
  #Etapa 1) Criando uma matriz com todas as coligações em ordem lexicográfica: 
  n = len(P[0]) #Número de critérios
  p = np.arange(1,n+1) #Vetor 1,2...n para ser utilizado como suporte para obter todas as combinações em ordem lexicográfica
  S = np.concatenate([np.pad(np.array(list(itertools.combinations(p, r))), ((0,0),(0,n-r)), 'constant', constant_values=0) for r in range(0,n+1)], axis=0)
    #^ Matriz que contem todas as combinações em ordem lexicográfica   

  #Etapa 2) Processo de ordenamento das alternativas em ordem crescente e coletando e ordenando seus índices 
  P_aumentado = np.pad(P, ((0,0),(0,1)), 'constant', constant_values=0)
  P_new = np.sort(P_aumentado)
  P_indice = np.argsort(P) + 1  

  M_t = np.zeros((len(P), 2**n)) 

  #Etapa 3) Cálculo da Integral de Choquet
  for i in range(0,len(P)): 
    for j in range(1, n+1): 
      P_indice_aux = np.pad(np.sort(P_indice[i][j-1:n]), ((0,j-1)), 'constant', constant_values=0)
      M_t[i][np.flatnonzero(np.all(S== P_indice_aux, 1))] = P_new[i][j] - P_new[i][j-1] 

  M_t  #Matriz de interesse transposta
  return M_t

"""**Inserindo os dados para criar as matrizes e aplicar no solver**"""

#Gerando dados aleatórios:

n_alter = 1000
n_crit = 4
tamanho = n_alter*n_crit
P = np.random.randint(0,100,size=tamanho).reshape(n_alter,n_crit)/100 #Matriz de decisão 

n = len(P[0]) #Número de critérios
capacidade = np.sort(np.random.uniform(0,100,size=2**n)/100) #Estratégia para gerar medidas que respeitam a monoticidade.
capacidade = np.random.uniform(0,100,size=2**n)/100 #Capacidades que não respeitam a monotocidade para testar se o solver esta funcioando corretamente.
capacidade[0] = 0
capacidade[2**n -1] = 1

M_t = matrizM(P)
utilidade = np.dot(M_t, capacidade) #Valor da Integral de Choquet.

e = np.random.normal(0, 0.1, n_alter) #Adicionando Ruído 
utilidade = np.dot(M_t, capacidade) + e #Valor da Integral de Choquet com ruído.

#print(P)
#print(capacidade)
#print(utilidade)

"""**2) Etapa: Obtenção dos demais dados de entrada (matrizes) necessários para o solver:**
**Matriz Q:** 2*Somatório de i até n do produto entre M[i].M_transposta[i];
**Matriz f:** somatório de i até n de -2.Produto entre ui.M[i];
**Matriz R:** Matriz das restrições de dimensão (m(2^(m-1)-1)+4 x 2^m);
**Matris s:** Matriz de restrições de dimensão (m(2^(m-1)-1)+4);
*OBS: n = número de alternativas; m = número de critérios.*
"""

#2° Etapa) Obter os demais dados de entrada (matrizes):

n = len(P[0]) #Número de critérios
M = M_t.transpose() #Matriz M

#i) Obtendo a Matriz Q
Q = 2*np.dot(M,M_t)

#ii) Obtendo a Matriz f 
f_transp = -2*np.dot(utilidade, M_t).reshape((2**n,))

#iii) Obtendo a Matris s
s = np.zeros(n*(2**(n-1)-1)+4)
s[n*(2**(n-1)-1)+3] = -1
s[n*(2**(n-1)-1)+2] = 1
s_t = s.transpose()

#iv) Contruindo a matriz R
R_oficial = np.zeros((n*(2**(n-1)-1), 2**n)) 
p = np.arange(1,n+1)
S = np.concatenate([np.pad(np.array(list(itertools.combinations(p, r))), ((0,0),(0,n-r)), 'constant', constant_values=0) for r in range(0,n+1)], axis=0) #Matriz das combinações em ordem lexicográfica que serve de base para construir a matriz R.

  #Construindo as 2 primeiras e as duas últimas linhas:
a = np.zeros((2, 2**n))
a[0][0] = 1
a[1][0] = -1
c = np.zeros((2, 2**n))
c[0][2**n- 1] = 1
c[1][2**n -1] = -1

  #Obtendo as posições onde os índices vão ser -1:
indice_menos1 = []
for i in range(2, n+1):
  teste_x = np.array(list(itertools.combinations(p, i)))
  #print(teste_x)

  for r in range(0, len(np.array(list(itertools.combinations(p, i-1))))):
    for h in range(0, len(teste_x)):
      if sum(np.in1d(teste_x, np.array(list(itertools.combinations(p, i-1)))[r]).reshape(teste_x.shape)[h]) == len(np.array(list(itertools.combinations(p, i-1)))[0]):
        primeiro_resultado_x = np.array(h)
        #print(primeiro_resultado_x)

        segundo_resultado_x = np.flatnonzero(np.all(S == np.pad(teste_x, ((0,0),(0,n-i)), 'constant', constant_values=0)[primeiro_resultado_x] , 1))  
        #print(segundo_resultado_x)

        indice_menos1 += [segundo_resultado_x]

indice_menos1 = np.concatenate((np.array(indice_menos1)), axis = None) #Deixando na horizontal
#print(indice_menos1)

  #Obtendo as posições em que os índices vão ser +1:
valor_1 = [np.sort(np.concatenate(([np.flatnonzero(np.all(S == np.pad(np.array(list(itertools.combinations(p, i-1))), ((0,0),(0,n-(i-1))), 'constant', constant_values=0)[r] , 1)) for r in range(0, len(np.array(list(itertools.combinations(p, i-1)))))]*(n-(i-1))), axis=None)) for i in range(2,n+1)]
valor_1_conc = np.concatenate((valor_1), axis = None)
#print(valor_1_conc)

#Substituindo na matriz das restrições:
for k in range(0, n*(2**(n-1)-1)):
  R_oficial[k][valor_1_conc[k]] = 1
  R_oficial[k][indice_menos1[k]] = -1
#print(R_oficial)

#Por fim:
print(" ")
R = np.concatenate([a,R_oficial,c])

#print(M)
#print(Q)
#print(f_transp)
#print(s_t)
#print(R)

"""**3) Etapa: Aplicando os dados em um solver**

**Solver utilizado: Scipy.Optmize (metodo Trust-Region Constrained Algorithm - 'trust-constr')**
"""

from scipy.optimize import minimize

#1) Definir a função objetivo:
def fun(x):
  return (0.5*np.dot(np.dot(x.transpose(),Q), x)) + (np.dot(f_transp,x))

#2) Palpite inicial
x0 = np.sort(np.random.uniform(0,100,size=2**n)/100)
x0[0] = 0
x0[(2**n)-1] = 1
print("Palpite incial:", x0)
print(" ")

#3) Colocando as fronteiras
from scipy.optimize import Bounds
lb = np.zeros(2**n)
ub = np.ones(2**n)
bounds = Bounds(lb, ub)

#4) Colocando as restrições
  #Restrições lineares:
from scipy.optimize import LinearConstraint
linear_constraint = LinearConstraint(R, -np.inf , s_t)

#5) Rodando no solver
res = minimize(fun, x0, method='trust-constr', 
               options={'verbose': 1},
               constraints=linear_constraint,
               bounds=bounds)

print(res.x)
print(" ")
#Validação:
Choquet_Integral = np.dot(M_t, res.x)
print("Valor da Integral de Choquet:", Choquet_Integral)

"""**Identificação das Capacidades de Choquet no domínio do índice de interação**

**Preciso criar a matriz de transformação T**
"""

#Calculando uma combinação:
import operator as op
from functools import reduce

#Definindo uma função para calculo das combinação (q,r):
def ncr(q, p):
    r = min(q, q-p)
    numer = reduce(op.mul, range(q, q-p, -1), 1)
    denom = reduce(op.mul, range(1, p+1), 1)
    return numer // denom  #or / in Python 2

from scipy.special import bernoulli #Biblioteca para importar os Números de Bernoulli.
B = bernoulli(n) #Números de Bernoulli
Brq = np.zeros((n+1, n+1)) #Matriz de suporte, inicialmente zerada

for r in range(0, n+1):
    for q in range(0, r+1):
      a = sum([ncr(q,p)*B[r-p] for p in range(0,r+1)])
      Brq[q][r] = a

#Construindo, definitivamente, a Matriz de Transformação T
  #1) Valores de r, na Matriz de Transformação T:
r_vector = []
for i in range(1, n+1):
  teste_x = np.array(list(itertools.combinations(p, i)))
  for i in range(0, len(teste_x)):
    r_vector += [len(teste_x[i])]
#print(r_vector)
r = np.pad(r_vector, ((1,0)), 'constant', constant_values=0)

  #2) Valores de q, na Matriz de Transformação T:
S_1 = [np.array(list(itertools.combinations(p, r))) for r in range(1,n+1)]
q_vector = []
for i in range(0, n):
  for j in range(0, len(S_1[i])):
    for k in range(0, n):
      for l in range(0, len(S_1[k])):
        q_vector += [sum(np.in1d(S_1[k], S_1[i][j]).reshape(S_1[k].shape)[l])]

q_final = np.reshape(q_vector, (2**n -1, 2**n -1))
q = np.pad(q_final, ((1,0),(1,0)), 'constant', constant_values=0)

  #3) Subistituindo os valores corretos nas posições corretas na Matriz de Transformação:
T = np.zeros((2**n,2**n)) 
for i in range(0, 2**n):
  for j in range(0, 2**n):
    T[i][j] = Brq[q[i][j]][r[j]]
T #Matriz de Transformação pronta

#print(Brq)
#print(r) 
#print(q) 
#print(T)

#Obtendo as matrizes/dados de entrada do Solver para o Domínio do Índice de Interação:

Q_i = np.dot(np.dot(T.transpose(),Q), T)
f_transp_i = np.dot(f_transp, T).reshape((2**n,))
R_i = np.dot(R, T)

s = np.zeros(n*(2**(n-1)-1)+4)
s[n*(2**(n-1)-1)+3] = -1
s[n*(2**(n-1)-1)+2] = 1
s_t = s.transpose()

#print(Q_i)
#print(f_transp_i)
#print(R_i)
#print(s_t)

"""**Solver no domínio do Índice de Interação**"""

from scipy.optimize import minimize

#1) Definir a função objetivo:
def fun(x):
  return (0.5*np.dot(np.dot(x.transpose(),Q_i), x)) + (np.dot(f_transp_i,x))

#2) Palpite inicial
x0 = np.sort(np.random.uniform(0,100,size=2**n)/100) #Acho que esse palpite inicial não faz mais tanto sentido!
x0[0] = 0
x0[(2**n)-1] = 1
print("Palpite incial:", x0)
print(" ")

#4) Colocando as restrições
  #Restrições lineares:
from scipy.optimize import LinearConstraint

linear_constraint = LinearConstraint(R_i, -np.inf , s_t)

# 5) Rodando
  #Somente restrições
    #OBS: Cada method tem sua própia maneira de definir as restrições, então, dependendo de qual metodo você deseje utilizar, você tem que adaptar as restrições
res = minimize(fun, x0, method='trust-constr', 
               options={'verbose': 1, 'maxiter':2000},
               constraints=linear_constraint)

print(res.x)
print(" ")
#Validação:
print("Voltando p/ o domínio das capacidades de Choquet:")
print(np.dot(T, res.x))
