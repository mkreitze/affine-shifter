import numpy as np
import matplotlib.pyplot as plt
import itertools as it
VERBOSE = False
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


# Brute force time
def bruteForceCheck(alphabet = [0,1],windowSize = "2x2", half = False):
    allRs = makeAllRshifts(alphabet)
    allDs = makeAllDshifts(alphabet)
    validShifters = np.zeros((len(allRs),len(allDs)))
    print(validShifters.shape)
    idxR = 0
    idxD = 0
    idxV = 0
    outputs = open(f"Valid Shift Arrays {windowSize} a{len(alphabet)} Half{half}.txt","w")
    for rShift in allRs:
        for dShift in allDs:
            RD = np.matmul(rShift,dShift)
            RD = RD % len(alphabet)
            DR = np.matmul(dShift,rShift)
            DR = DR % len(alphabet)
            if half or idxD<=idxR:
                if np.array_equal(RD,DR):
                    validShifters[idxR,idxD] = 1
                    outputs.write("Valid pair: \n")
                    outputs.write(f"R shifter:{idxR} \n")
                    np.savetxt(outputs,rShift, fmt = '%d')
                    outputs.write(f"D shifter:{idxD} \n")
                    np.savetxt(outputs,dShift, fmt = '%d')
                    idxV += 1
            idxD += 1
        idxD = 0
        idxR += 1
    outputs.write(f"Number of valid shift pairs: {idxV} \n")
    plt.matshow(validShifters)
    plt.colorbar()
    plt.savefig(f"Valid Shifters {windowSize} a{len(alphabet)} Half{half}", dpi=300)
    return()

bruteForceCheck(alphabet = [0,1,2],windowSize = "2x2", half = False)

# find R^a = I = D^b
def produceDeBT():
    return()