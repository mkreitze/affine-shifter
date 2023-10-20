import numpy as np
import matplotlib.pyplot as plt
import itertools as it
VERBOSE = False # Prints out some reporting stats (not implemented properly)
RECORDCOMMUTERS = True # Records the DR = RD matricies 
HALF = False # True -> Use half
WINDOWSIZE = "2x2" # Not implemented for general windows, currently hardcoded for 2x2
ALPHABET = [0,1] # enumerate alphabet, this is needed for determinant conditions
ALPHABETSIZE = len(ALPHABET)
OUTPUT = f"deBTShifters 2x2 a{len(ALPHABET)} Half{HALF}.txt"
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
# Actually cant use this nvm 
def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

# Determinant conditions for shifters (sub matrix det == 0), can be generalized from 2x2 case easily
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

# Affine condition on R/D matricies, can be generalized from 2x2 case easily
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
# There must be a better way to enumerate all possible matricies. Must think for future cases
def makeAllRshifts(alphabet = [0,1]):
    rMats = []
    rMat = initRShifter()
    detConditions = makeDetConds(alphabet)
    consistConditions = makeConstConds(alphabet)
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
            for c in range(2**4): #size 2 alphabet, 4 unresolved entries
                binC = "{0:b}".format(c)
                binC = (4-len(binC))*"0" + binC # resolves not enough digits 
                rMat[2,2] = int(binC[0])  
                rMat[2,3] = int(binC[1])
                rMat[3,2] = int(binC[2])
                rMat[3,3] = int(binC[3])
                newMat = np.copy(rMat)
                rMats.append(newMat)
    return(rMats)

    

# Makes all the Dshifts and put them to a list
# There must be a better way to enumerate all possible matricies. Must think for future cases
def makeAllDshifts(alphabet=[0,1]):
    dMats = []
    dMat = initDShifter()
    detConditions = makeDetConds(alphabet)
    consistConditions = makeConstConds(alphabet)
    # detConditions = np.matrix([[0,0,1,1],[0,1,1,1],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],]) # listed in binary smallest to largest to make pattern apparent later
    # consistConditions = np.matrix([[0,1],[1,0],[1,1]])
    # resolve  d21 d43 d41 d23
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
            for c in range(2**4): #size 2 alphabet, 4 unresolved entries
                binC = "{0:b}".format(c)
                binC = (4-len(binC))*"0" + binC # resolves not enough digits 
                dMat[1,1] = int(binC[0])  
                dMat[1,3] = int(binC[1])
                dMat[3,1] = int(binC[2])
                dMat[3,3] = int(binC[3])
                newMat = np.copy(dMat)
                dMats.append(newMat)
    return(dMats)

def makeDRRD(R,D):
    RD = np.matmul(R,D)
    RD = RD % ALPHABETSIZE # Makes DR and RD
    DR = np.matmul(D,R)
    DR = DR % ALPHABETSIZE
    return(RD,DR)

# Brute force time
# Generates all possible R/D shifters
# Checks if DR = RD ( commutable )
# Checks for m,n | R^m = D^n = I and m*n = |A|^{windowSize}
# Checks R^i*D^j != I for all i,j < m,n

# Later results found symmtry, that is if RnDm = DmRn then RmDn = DnRm. This means we only need to check half the space.
# Need to check 
def bruteForceComCheck():
    allRs = makeAllRshifts(ALPHABET)
    allDs = makeAllDshifts(ALPHABET)
    if VERBOSE:
        print(f"Possible shifters:({len(allRs)},{len(allDs)})") 
    validGraphDs = [];validGraphRs = []
    validDs = [];validRs = []
    idxR = 0;idxD = 0;idxV = 0
    commuterOutput = open(f"CommuterPairs {WINDOWSIZE} a{ALPHABETSIZE} Half{HALF}.txt","w")
    for rShift in allRs:
        for dShift in allDs:
            if not(HALF) or idxD<=idxR:
                RD,DR = makeDRRD(rShift,dShift)
                if np.array_equal(RD,DR): 
                    if RECORDCOMMUTERS:                    
                        validGraphDs.append(idxD);validGraphRs.append(idxR)
                        commuterOutput.write("Valid pair: \n")
                        commuterOutput.write(f"R shifter:{idxR} \n");np.savetxt(commuterOutput,rShift, fmt = '%d')
                        commuterOutput.write(f"D shifter:{idxD} \n");np.savetxt(commuterOutput,dShift, fmt = '%d')
                    validDs.append(dShift);validRs.append(rShift)
                    idxV += 1
            idxD += 1
        idxD = 0
        idxR += 1
    if RECORDCOMMUTERS:
        plt.scatter(validGraphDs, validGraphRs,s=0.5)
        plt.savefig(f"ShiftPairs {WINDOWSIZE} a{ALPHABETSIZE} Half{HALF}.png", dpi=300)
        plt.xlabel("D Shifters")
        plt.ylabel("R Shifters")
    return(validDs,validRs)

# given a valid shifter pair, we find R^m = R and D^n = D
def determinePower(M,maxSize,alphabet,dim):
    idx = 2
    I = np.identity(dim)
    A = np.copy(M)
    while idx < maxSize:
        A = A @ M
        A = A % alphabet
        if np.array_equal(A,I):
            break
        idx += 1
    if idx == maxSize:
        return(0)
    return(idx)

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

# Determines all commutative shifters
comDs,comRs = bruteForceComCheck()
# Now determine powers and generate deBT
output = open(OUTPUT,'w')
for D,R in zip(comDs,comRs): # walks through all valid shifters
    n = determinePower(R,81,ALPHABETSIZE,5)
    m = determinePower(D,81,ALPHABETSIZE,5)
    if m*n == 16: # make general, 16 from binary alphabet on 2x2 window
        output.write(f"R shifter, power {m}: \n")
        np.savetxt(output,R, fmt = '%d')
        output.write(f"D shifter, power {n}: \n")
        np.savetxt(output,D, fmt = '%d')
        deBT = makeDeBT(R,D,n,m,INITWINDOW,ALPHABETSIZE)
        output.writelines(str(deBT)+"\n")
        output.writelines(str(R@R@D@D%2)+"\n")
