import numpy as np
import sympy as sp
import math

paramfile='param.txt'
Pkfile='Pk.txt'

#reading parameter file-----------
par,val=np.genfromtxt(paramfile,usecols=(0,1),dtype='str',unpack=True)

log2s=float(val[par=='log2size'])
formula=val[par=='Pk']
probing=float(val[par=='Pk_probing'])
precision=tuple(val[par=='Pk_precision'].tolist())[0]
#---------------------------------


maxk=2**log2s* 1.415 +1 #upper boundary for k
n=math.ceil(probing * maxk) #number of datapoints for P(k)


#creating function from string expression
def create_function(expression):
    k = sp.symbols('k')
    expr = sp.sympify(expression)
    return sp.lambdify(k, expr, modules=["numpy"])


func = create_function(formula)

#making k,P(k)
PK=np.zeros((n,2))
kvals=np.linspace(0.,maxk,n) #k values
kvals[0]=kvals[1] #to avoid dividing by 0

PK[:,0]=kvals
PK[:,1]=func(kvals)
PK[0,0]=0. #giving 0 back

np.savetxt(Pkfile,PK,fmt=precision)

print(f'Power spectrum listed in {Pkfile}\n')