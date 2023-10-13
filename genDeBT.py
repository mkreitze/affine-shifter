import numpy as np
import matplotlib.pyplot as plt
import itertools as it

ALPHABET = 2
WINDOWSIZE = 4
INPUT = "ShiftPairs 2x2 a2 HalfFalse.txt"
OUTPUT = "deBTShifters 2x2 a2 HalfFalse.txt"
INITWINDOW = np.array([0,0,0,0,1],dtype = int)
INITWINDOW.shape = (5,1)
VERBOSE = False


# given a valid shifter pair, we find R^m = R and D^n = D
def determinePower(M,maxSize,alphabet,dim):
    idx = 2
    I = np.identity(dim)
    while idx < maxSize:
        M = np.dot(M,M)
        M = M % alphabet
        if np.array_equal(M,I):
            break
        idx += 1
    if idx == maxSize:
        return(0)
    return(idx)

# Gets data from previous program
def pullAllFromText(file,arrayDim = 5):
    allLines = file.readlines()
    allRs = []
    allDs = []
    block = 2*arrayDim+3 
    # Gets all the R shifter matricies
    for idx in range(len(allLines)):
        newArray = []
        if idx%block == 2:
            lines = allLines[idx:idx+arrayDim]
            for line in lines:
                newArray.append(line.split())
            allRs.append(np.array(newArray,dtype = int))
    # Gets all the D shifter matricies
    for idx in range(len(allLines)):
        newArray = []
        if idx%block == 3+arrayDim:
            lines = allLines[idx:idx+arrayDim]
            for line in lines:
                newArray.append(line.split())
            allDs.append(np.array(newArray,dtype = int))

    return(allRs,allDs)

def makeDeBT(R,D,n,m,initWindow,alphabetSize):
    # Generate deBT:
    firstInRow = np.copy(initWindow)
    deBT = np.zeros((n,m),dtype=int)
    for i in range(n):
        nextInCol = np.copy(firstInRow)
        for j in range(m):
            deBT[i,j] = nextInCol[0,0]
            nextInCol = R @ nextInCol
            nextInCol = nextInCol % alphabetSize
        firstInRow = D @ firstInRow
        firstInRow = firstInRow % alphabetSize
    return(deBT)


file = open(INPUT,'r')
output = open(OUTPUT,'w')
allRs,allDs = pullAllFromText(file)
for R,D in zip(allRs,allDs): # walks through all valid shifters
    n = determinePower(R,16,2,5)
    m = determinePower(D,16,2,5)
    if m*n == 16: # make general, 16 from binary alphabet on 2x2 window
        output.write(f"R shifter, power {m}: \n")
        np.savetxt(output,R, fmt = '%d')
        output.write(f"D shifter, power {n}: \n")
        np.savetxt(output,D, fmt = '%d')
        deBT = makeDeBT(R,D,n,m,INITWINDOW,ALPHABET)
        output.writelines(str(deBT)+"\n")
