import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import itertools as it
import jax.numpy as jnp
import jax as j
import math as m   
import galois
SANITY = False # Takes two found valid R and D shifters and shows it goes through the program correctly
REPORTSHIFTERS = True # Records all valid matricies
REPORTTORI = True # Records all unique deBT found
PLOT = True # Plots barplots of powers
MAIN = True # Incase I want to use code in here elsewhere
WINDOWSIZE = "2x2" # Not implemented for general windows, currently hardcoded for 2x2
NUMOFCELLS = 4 # Note, each R/D matrix is a square matrix of this + 1
ALPHABET = [0,1] # enumerate alphabet, this is needed for determinant conditions

OUTPUT = f"deBTShifters 2x2 a{len(ALPHABET)}.txt"
ALPHABETSIZE = len(ALPHABET)
INITWINDOW = np.array([1,1,0,1,1],dtype = int)
INITWINDOW.shape = (5,1)
GF = galois.GF(ALPHABETSIZE)

# with j.default_device(j.devices("cpu")[0]): # forces cpu cause cuda is annoying (not using jax rn)

# Makes default Rshift

def initRShifter():
    rMat = np.zeros((5,5),dtype=int) 
    rMat[0,2] = 1
    rMat[1,3] = 1
    rMat[4,4] = 1
    if SANITY:
        print("Generic R shifter")
        print(rMat)
    return(rMat)

# Makes default DShift
def initDShifter():
    dMat = np.zeros((5,5),dtype=int) 
    dMat[0,1] = 1
    dMat[2,3] = 1
    dMat[4,4] = 1
    if SANITY:
        print("Generic D shifter")
        print(dMat)
    return(dMat)

# Determinant conditions for shifters (sub matrix det == 0), currently for arbtirary alphabet on the 2x2 case s
# Should be able to generalize from 2x2 window case
def makeDetConds(alphabet):
    windowSize = 4
    windows = it.product(alphabet,repeat = windowSize)
    validWindows = []
    for window in windows:
        if ((window[0]*window[1] - window[2]*window[3])%len(alphabet)) != 0:
            validWindows.append(window)
    npWindows = np.array(validWindows)
    if SANITY:
        print("Valid sub matricies [a d b c]:")
        print(npWindows)
    return(npWindows)

# Affine condition on R/D matricies, currently for arbitrary alphabet on the 2x2 case
# Should be able to generalize from 2x2 window case easily
def makeConstConds(alphabet):
    windowSize = 2
    windows = it.product(alphabet,repeat = windowSize)
    validWindows = []
    for window in windows:
        if (window[0] != 0) or (window[1] != 0):
            validWindows.append(window)
    npWindows = np.array(validWindows)
    if SANITY:
        print("Valid final col options")
        print(npWindows)
    return(npWindows)

# Makes all the Rshifts and put them to a list
# There must be a better way to enumerate all possible matricies. Currently just go through all possibilities in binary
def makeAllRshifts(alphabet):
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
def makeAllDshifts(alphabet):
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
# Stores M^1 to M^m ( I stored for later deBT checking )
def determinePower(M,maxSize,alphabetSize,dim,bool = False):
    powRec = [] #records all the powers... useful for later
    N = np.copy(M)
    powRec.append(N)
    I = np.eye(dim)
    for i in range(maxSize-2):
        N = N @ M
        N = N % alphabetSize
        if np.array_equal(N,I):
            powRec.append(np.copy(N))
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

# Stolen from user Sublow at https://stackoverflow.com/questions/47064885/list-all-factors-of-number
def factorize(num):
    return [n for n in range(1, num + 1) if num % n == 0]

# Correct Method Not used, incorrect somehow
def checkCrossTerms(rPows,dPows,alphabetSize):
    dims = np.shape(rPows[0][0])
    dimA = dims[0]-1
    for rP in rPows:
        for dP in dPows:
            temp = GF((rP - dP) % alphabetSize)
            A = temp[0:dimA,0:dimA] # for readability
            Ab = temp[0:dimA,0:dims[0]] # for readability
            nullA = A.null_space()
            nullAb = Ab.null_space()
            if (nullA.shape[0] != nullAb.shape[0]):
                return(False)
    return(True)

def plotPows(pows,name,factors,alphabetSize):
    figR = plt.figure().gca()
    ys = [] 
    xs = list(range(len(pows)))
    # xTicks = np.append(fig.get_xticks(),factors)
    for powerQuant in pows:
        ys.append(len(powerQuant))
    figR.xaxis.set_major_locator(MaxNLocator(integer = True)) # forces axis to be integers
    bars = plt.bar(xs,ys,color = 'orange', width = 0.4)
    for fact in factors:
        bars[fact].set_color('red')
        height = bars[fact].get_height()
        figR.text( bars[fact].get_x() + bars[fact].get_width() / 2, height, f"{fact}", ha="center", va="bottom")
    plt.title(f"Distribution of {name} powers for |A| = {alphabetSize}")
    plt.savefig(f'Power Dist of {name} for A = {alphabetSize}',dpi = 400)   

# Hard coded for 2x2 window, need to think of how to do this in the future
# Takes in power matrix applied to some starting window
def checkConnections(allWindows):
    rFront = [2,3]; lFront = [0,1]
    dFront = [1,3]; uFront = [0,2]
    testR = np.roll(allWindows[:,:,:],-1,axis = 0) 
    testD = np.roll(allWindows[:,:,:],-1,axis = 1) 
    if np.array_equal(testR[:,:,lFront],allWindows[:,:,rFront]): # notice R needs all things 'sent left' to match with everything 'sent right'
        if np.array_equal(testD[:,:,uFront],allWindows[:,:,dFront]): # notice D needs all things 'sent up' to match with everything 'sent down'
            return(True) 
    else:
        return(False)

def checkUniqueness(allWindows,alphabetSize,initWindow,numOfCells):
    converter = np.array(alphabetSize**(np.arange(numOfCells,dtype=int)).reshape(numOfCells,1))
    w = len(initWindow) - 1
    converted = allWindows[:,:,0:w,0:w,np.newaxis] @ converter[:,:,np.newaxis]
    sums = np.sum(converted,axis = 2)
    extract = np.sort(sums.flatten())
    allNums = np.arange(0,alphabetSize**w)
    return(np.array_equal(allNums,extract))





# Brute force checking
# Walk through all possible  
def bruteForceSearch(maxSize,dim,alphabet,alphabetSize,initWindow,numOfCells):
    # R shift pairs that make deBT with 0000 initial. Used for checking
    if SANITY:
        RGood = np.array([[0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [1, 0, 1, 1, 1],
        [0, 0, 0, 0, 1]])
        DGood = np.array([[0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 1]])

    allRs = makeAllRshifts(alphabet)
    allDs = makeAllDshifts(alphabet)
    # by defintion the lists that sort by power include 0, this is the 'trash' that cannot make deBT as their power is bigger than max.
    rPowRecs = [[] for i in range(maxSize+1)]; dPowRecs = [[] for i in range(maxSize+1)] # Records all R,D powers which have correct dimension
    output = open(f"{OUTPUT}","w")
    validShifters = [[],[],[],[]] # stores as R matrix, D matrix, R^m = I power, D^n = I power
    # Sort by the shifters by their cyclic powers
    for rShift in allRs:
        i,rec = determinePower(rShift,maxSize,alphabetSize,dim)
        if SANITY: # Checks to make sure a proper pair goes through
            if np.array_equal(RGood,rShift):
                print("Indicies for solution R when stored")
                print(i,len(rPowRecs[i]))
        rPowRecs[i].append(rec)
    for dShift in allDs:
        i,rec = determinePower(dShift,maxSize,alphabetSize,dim)
        if SANITY: # Checks to make sure a proper pair goes through
            if np.array_equal(DGood,dShift):
                print("Indicies for solution D when stored")
                print(i,len(dPowRecs[i]))
        dPowRecs[i].append(rec)
    # Check for commutability
    # This is done smartly, only R/D matrix pairs with matching factors as cyclic matrix powers
    factors = factorize(maxSize) 
    halfway = m.ceil(len(factors)/2)
    if PLOT: # turn into its own function, just generates bar plots. these explain why our method fails for higher |A| cases
        plotPows(rPowRecs,"R",factors,alphabetSize)
        plotPows(dPowRecs,"D",factors,alphabetSize)    
    idxSuccess = 0
    for i in range(halfway):
        rIndex = factors[i] # Is horz dimension of the tori
        dIndex = factors[len(factors)-i-1] # Is vert dimension of the tori
        idxR=0 # Is specific R with cyclic power rIndex
        for R in rPowRecs[rIndex]:
            idxD=0 # Is specific D with cyclic power dIndex
            for D in dPowRecs[dIndex]:
                # RD = (R[0] @ D[0]) % alphabetSize
                # DR = (D[0] @ R[0]) % alphabetSize
                # if np.array_equal(RD,DR):
                Rpows = np.array(rPowRecs[rIndex][idxR]); Dpows = np.array(dPowRecs[dIndex][idxD])
                allPows = np.matmul(Rpows[:, np.newaxis],Dpows[np.newaxis,:])
                allWindows = (allPows @ initWindow) % alphabetSize
                if checkUniqueness(allWindows,alphabetSize,initWindow,numOfCells):
                    if checkConnections(allWindows):
                        idxSuccess+=1
                        validShifters[0].append(R)
                        validShifters[1].append(D)
                        validShifters[2].append(rIndex)
                        validShifters[3].append(dIndex)
                    if REPORTSHIFTERS:
                        output.write(f"Valid pair, dimension {rIndex} x {dIndex} \n")
                        output.write(f"R is {idxR} in list D is {idxD} in list. Pair is number {idxSuccess}. \n")
                        output.write(str(R[0]))
                        output.write('\n')
                        output.write(str(D[0]))
                        output.write('\n')
                        output.write(str(makeDeBT(R[0],D[0],rIndex,dIndex,initWindow,alphabetSize)))
                        output.write('\n')
                    if SANITY:
                        if np.array_equal(RGood,R[0]):
                            if np.array_equal(DGood,D[0]):
                                print("Indicies for solution R when pulling out")
                                print(rIndex,idxR)
                                print("Indicies for solution D when pulling out")
                                print(dIndex,idxD)
                                print("Solution R,D commutable?")
                                print(np.array_equal(RD,DR))
                idxD+=1
            idxR+=1
    return(validShifters)
    
# Gets data from previous program, old code that needs to be fixed up to be used
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

if MAIN:
    # "Main" below
    maxToriSize = ALPHABETSIZE**NUMOFCELLS
    shifterDimension = NUMOFCELLS+1
    bruteForceSearch(maxToriSize,shifterDimension,ALPHABET,ALPHABETSIZE,INITWINDOW,NUMOFCELLS)

