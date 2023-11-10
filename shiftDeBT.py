import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import math
VERBOSE = False # Prints out some reporting stats (not implemented properly)
REPORT = True # Records the DR = RD matricies 
PLOT = True
WINDOWSIZE = "2x2" # Not implemented for general windows, currently hardcoded for 2x2
NUMOFCELLS = 4 # Note, each R/D matrix is a square matrix of this + 1
ALPHABET = [0,1] # enumerate alphabet, this is needed for determinant conditions
ALPHABETSIZE = len(ALPHABET)
OUTPUT = f"deBTShifters 2x2 a{len(ALPHABET)}.txt"
INITWINDOW = np.array([0,0,0,0,1],dtype = int)
INITWINDOW.shape = (5,1)

# Makes default Rshift
def initRShifter():
    rMat = np.zeros((5,5)) 
    rMat[0,2] = 1
    rMat[1,3] = 1
    rMat[4,4] = 1
    if VERBOSE:
        print(rMat)
    return(rMat)


# Makes default DShift
def initDShifter():
    dMat = np.zeros((5,5)) 
    dMat[0,1] = 1
    dMat[2,3] = 1
    dMat[4,4] = 1
    if VERBOSE:
        print(dMat)
    return(dMat)

def sanityCheck():
    sanityCheckR = initRShifter()
    print(sanityCheckR)

    sanityCheckD = initDShifter()
    print(sanityCheckD)

    a = np.matmul(sanityCheckR,sanityCheckD)
    print(a)
    b = np.matmul(sanityCheckD,sanityCheckR)
    print(b)
    if np.array_equal(a,b) == True:
        print(True)

# Thank you https://stackoverflow.com/questions/2267362/how-to-convert-an-integer-to-a-string-in-any-base/28666223#28666223
# Use this in the future
def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

# Determinant conditions for shifters (sub matrix det == 0), currently for arbtirary alphabet on the 2x2 case 
# Should be able to generalize from 2x2 window case
def makeDetConds(alphabet = [0,1]):
    windowSize = 4
    windows = it.product(alphabet,repeat = windowSize)
    validWindows = []
    for window in windows:
        if ((window[0]*window[1] - window[2]*window[3])%len(alphabet)) != 0:
            validWindows.append(window)
    npWindows = np.array(validWindows)
    if VERBOSE:
        print(npWindows)
    return(npWindows)

# Affine condition on R/D matricies, currently for arbitrary alphabet on the 2x2 case
# Should be able to generalize from 2x2 window case easily
def makeConstConds(alphabet = [0,1]):
    windowSize = 2
    windows = it.product(alphabet,repeat = windowSize)
    validWindows = []
    for window in windows:
        if (window[0] != 0) or (window[1] != 0):
            validWindows.append(window)
    npWindows = np.array(validWindows)
    if VERBOSE:
        print(npWindows)
    return(npWindows)


# Makes all the Rshifts and put them to a list
# There must be a better way to enumerate all possible matricies. Currently just go through all possibilities in binary
def makeAllRshifts(alphabet = [0,1]):
    rMats = []
    rMat = initRShifter()
    detConditions = makeDetConds(alphabet) #pre-calcs valid det = 0 possibilities
    consistConditions = makeConstConds(alphabet) #pre-calcs valid 0 col possiblities
    # resolve r31 r42 r41 r32
    for a in range(detConditions.shape[0]): 
        rMat[2,0] = detConditions[a,0]
        rMat[3,1] = detConditions[a,1]
        rMat[3,0] = detConditions[a,2]
        rMat[2,1] = detConditions[a,3]
        # resolve r35 r45
        for b in range(consistConditions.shape[0]): 
            rMat[2,4] = consistConditions[b,0]
            rMat[3,4] = consistConditions[b,1]
            # resolve r33 r34 r43 r44
            for c in range(ALPHABETSIZE**NUMOFCELLS): #using global variables, sue me 
                # binC = "{0:b}".format(c)
                # binC = (4-len(binC))*"0" + binC # resolves not enough digits 
                binC = np.base_repr(c,ALPHABETSIZE)
                binC = (4-len(binC))*"0" + binC # resolves not enough digits                 
                rMat[2,2] = int(binC[0])  
                rMat[2,3] = int(binC[1])
                rMat[3,2] = int(binC[2])
                rMat[3,3] = int(binC[3])
                newMat = np.copy(rMat)
                rMats.append(newMat)
    return(rMats)

    

# Makes all the Dshifts and put them to a list
# There must be a better way to enumerate all possible matricies. Currently just go through all possibilities in binary
def makeAllDshifts(alphabet=[0,1]):
    dMats = []
    dMat = initDShifter()
    detConditions = makeDetConds(alphabet) #pre-calcs possibilities
    consistConditions = makeConstConds(alphabet) 
    for a in range(detConditions.shape[0]): 
        dMat[1,0] = detConditions[a,0]
        dMat[3,2] = detConditions[a,1]
        dMat[3,0] = detConditions[a,2]
        dMat[1,2] = detConditions[a,3]
        # resolve d25 d45
        for b in range(consistConditions.shape[0]): 
            dMat[1,4] = consistConditions[b,0]
            dMat[3,4] = consistConditions[b,1]
            # resolve d22 d24 d42 d44
            for c in range(ALPHABETSIZE**NUMOFCELLS): #using global variables, sue me 
                # binC = "{0:b}".format(c)
                # binC = (4-len(binC))*"0" + binC # resolves not enough digits 
                binC = np.base_repr(c,ALPHABETSIZE)
                binC = (4-len(binC))*"0" + binC # resolves not enough digits                 
                dMat[1,1] = int(binC[0])  
                dMat[1,3] = int(binC[1])
                dMat[3,1] = int(binC[2])
                dMat[3,3] = int(binC[3])
                newMat = np.copy(dMat)
                dMats.append(newMat)
    return(dMats)

# Determines m | M^m = I
# Used for both R and D shifters
def determinePower(M,maxSize,alphabetSize,dim):
    powRec = [] #records all the powers... useful for later
    powRec.append(np.copy(M))
    N = np.copy(M)
    I = np.eye(dim)
    for i in range(maxSize):
        N = N @ M
        N = N % alphabetSize
        if np.array_equal(N,I):
            return(i+2,powRec) # +2, we start at 0 but in reality i = 0 produces the 2nd power
        powRec.append(np.copy(N))
    return(0,[])

# Generates a deBT from a pair of R,D shifters. Note you need to give the dimension
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


def checkCrossTerms(rPows,dPows,alphabetSize):
    I = np.eye(5)
    for rP in rPows:
        for dP in dPows:
            temp = (rP - dP) % alphabetSize
            det = np.linalg.det(temp)
            det = det % alphabetSize 
            if (det != 0):
                return(False)
    return(True)
# Brute force checking
# Walk through all possible  
def bruteForceSearch(maxSize,dim,alphabet,alphabetSize,initWindow):
    allRs = makeAllRshifts(alphabet)
    allDs = makeAllDshifts(alphabet)
    # by defintion the lists that sort by power include 0, this is the 'trash' that cannot make deBT as their power is bigger than max.
    rPows = [[] for i in range(maxSize+1)]; dPows = [[] for i in range(maxSize+1)]
    rPowsRecs = [[] for i in range(maxSize+1)]; dPowsRecs = [[] for i in range(maxSize+1)]
    if REPORT:
        output = open(f"{OUTPUT}.txt","w")

    validShifters = [[],[],[],[]] # stores as R matrix, D matrix, R^m = I power, D^n = I power
    # Sort by the shifters by their cyclic powers
    for rShift in allRs:
        i,rec = determinePower(rShift,maxSize,alphabetSize,dim)
        rPows[i].append(rShift)
        rPowsRecs[i].append(rec)
    for dShift in allDs:
        i,rec = determinePower(dShift,maxSize,alphabetSize,dim)
        dPows[i].append(dShift)
        dPowsRecs[i].append(rec)
    # Check for commutability
    # This is done smartly, only R/D matrix pairs which can even make tori are considered
    for i in range(maxSize-1):
        i = i+1
        j = math.floor(maxSize/i)
        if i > j:
            break
        if i*j == maxSize:
            idx1=0
            for R in rPows[i]:
                idx1+=1
                idx2=0
                for D in dPows[j]:
                    idx2+=1
                    RD = (R @ D) % alphabetSize
                    DR = (D @ R) % alphabetSize
                    if np.array_equal(RD,DR):
                         if checkCrossTerms(rPowsRecs[i][idx1],dPowsRecs[j][idx2],alphabetSize):
                            validShifters[0].append(R)
                            validShifters[1].append(D)
                            validShifters[2].append(i)
                            validShifters[3].append(j)       
                            if REPORT:
                                output.write(f"Valid pair, dimension {i} x {j} \n")
                                output.write(str(R))
                                output.write('\n')
                                output.write(str(D))
                                output.write('\n')
                                output.write(str(makeDeBT(R,D,i,j,initWindow,alphabetSize)))
                                output.write('\n')
    return()
    

# Gets data from previous program, not really needed
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



# Determines all commutative shifters
maxToriSize = ALPHABETSIZE**NUMOFCELLS
shifterDimension = NUMOFCELLS+1
bruteForceSearch(maxToriSize,shifterDimension,ALPHABET,ALPHABETSIZE,INITWINDOW)
# Now determine powers and generate deBT
