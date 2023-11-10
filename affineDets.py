import numpy as np
import itertools as it
import matplotlib.pyplot as plt

DIMS = [2]
ALPHABETS = [[0,1],[0,1,2],[0,1,2,3],[0,1,2,4]]
OUTPUT = f"affineInvertibles.txt"

def makeAllMatricies(dim,matrixSize,alphabetSize):
    seq = []
    for i in range(alphabetSize**(matrixSize)):
        a = np.base_repr(i,alphabetSize)
        a = (matrixSize-len(a))*"0" + a # resolves not enough digits
        A = np.array(list(a),dtype=int)
        A = A.reshape((dim,dim))
        det = (np.linalg.det(A)) % alphabetSize
        if det != 0:
            seq.append(i)
    return(seq)

output = open(f"{OUTPUT}","w")

# for alphabet in ALPHABETS:
#     pattern=[]
#     output.write(f"For alphabet {alphabet} \n")
#     for dim in DIMS:
#         matrixSize = dim*dim
#         alphabetSize = len(alphabet)
#         seq = makeAllMatricies(dim,matrixSize,alphabetSize)
#         pattern = pattern + seq
#         output.write(f"For dimension: {dim} \n")
#         output.write(f"{seq} \n")

#     xs = range(len(pattern))
#     plt.scatter(xs,pattern,s=1)
#     plt.savefig(f"Alphabet {alphabet} and Dims {DIMS}",dpi = 1000)

for dim in DIMS:
    pattern=[]
    output.write(f"For dim {dim} \n")
    for alphabet in ALPHABETS:
        matrixSize = dim*dim
        alphabetSize = len(alphabet)
        seq = makeAllMatricies(dim,matrixSize,alphabetSize)
        pattern = pattern + seq
        output.write(f"For dimension: {dim} \n")
        output.write(f"{seq} \n")

    xs = range(len(pattern))
    plt.scatter(xs,pattern,s=1)
    plt.savefig(f"Dim {dim} and alphabets {ALPHABETS}",dpi = 1000)
