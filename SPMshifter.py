import numpy as np
import itertools as it
import matplotlib.pyplot as plt
ALPHABET = [0,1] # enumerate alphabet, this is needed for determinant conditions
ALPHABETSIZE = len(ALPHABET)
WINDOWSIZE = "2x2" # This is just for file outputs, NUMOFCELLS actually affects the searcher
NUMOFCELLS = 4 # Note, each R/D matrix is a square matrix of this + 1
INITWINDOW = np.array(NUMOFCELLS*[0],dtype = int)
INITWINDOW.shape = (NUMOFCELLS,1)
INITWINDOW[0] = 1 # This produces the vec form of the initial window 1 0 0 ...  
SANITY = False


# Makes default Rshift
# Need to code in 1 entries for general case
def initRShifter():
    rMat = np.zeros((NUMOFCELLS,NUMOFCELLS),dtype=int) 
    rMat[0,2] = 1
    rMat[1,3] = 1
    if SANITY:
        print("Generic R shifter")
        print(rMat)
    return(rMat)

# Makes default DShift
# Need to code in 1 entries for general case
def initDShifter():
    dMat = np.zeros((NUMOFCELLS,NUMOFCELLS),dtype=int) 
    dMat[0,1] = 1
    dMat[2,3] = 1
    if SANITY:
        print("Generic D shifter")
        print(dMat)
    return(dMat)

# gets all possible entries that force det != 0
def getDetEntries(detSize,detShape):
    dets = []
    possDets = it.product(ALPHABET,repeat = detSize)
    for poss in possDets: 
        test2 = np.array(poss,dtype = int)
        test2.shape = detShape 
        if np.linalg.det(test2) != 0: # we only need to care for non zero despite this techincally being in a galois field
            dets.append(test2)
            if SANITY:
                print(test2)
    return(dets)


unresolvedEntries = 8

detSize = 4 # Generalize 
detRows = [2,3];detCols = [0,1];detShape = (2,2)
dets = getDetEntries(detSize,detShape)

freeEntries = unresolvedEntries-detSize 
allFrees = list(it.product(ALPHABET,repeat = freeEntries))
freeRows = [2,3];freeCols = [2,3];freeShape = (2,2)
# makes Rs 
# only applies det condition
rs = []
for det in dets:
    baseR = initRShifter()
    for frees in allFrees:
        freeChunk = np.array(frees,dtype = int)
        freeChunk.shape = freeShape
        baseR[detRows[0],detCols[0]:detCols[1]+1] = det[0]
        baseR[detRows[1],detCols[0]:detCols[1]+1] = det[1]
        baseR[freeRows[0],freeCols[0]:freeCols[1]+1] = freeChunk[0]
        baseR[freeRows[1],freeCols[0]:freeCols[1]+1] = freeChunk[1]
        rs.append(baseR)
    

# Generates a deBT from a pair of R,D shifters. Note you need to give the dimension
def makeDeBT(R,D,n,m):
    # Generate deBT:
    firstInRow = np.copy(INITWINDOW)
    deBT = np.zeros((n,m),dtype=int)
    for i in range(n):
        nextInCol = np.copy(firstInRow)
        for j in range(m):
            deBT[i,j] = nextInCol[0,0]
            nextInCol = R @ nextInCol
            nextInCol = nextInCol % ALPHABETSIZE
        firstInRow = D @ firstInRow
        firstInRow = firstInRow % ALPHABETSIZE
    unique, counts = np.unique(deBT, return_counts=True)
    uniqueCounts = dict(zip(unique, counts)) 
    if uniqueCounts[1] == 6:
        deBT[0] = 3
    return(deBT)


ds = []
detRows = [1,3];detCols = [0,1]
freeRows = [1,3];freeCols = [2,3];freeShape = (2,2)
# makes Ds 
# only applies det condition
for det in dets:
    baseD = initDShifter()
    for frees in allFrees:
        freeChunk = np.array(frees,dtype = int)
        freeChunk.shape = freeShape
        baseD[detRows[0],detCols[0]:detCols[1]+1] = det[0]
        baseD[detRows[1],detCols[0]:detCols[1]+1] = det[1]
        baseD[freeRows[0],freeCols[0]:freeCols[1]+1] = freeChunk[0]
        baseD[freeRows[1],freeCols[0]:freeCols[1]+1] = freeChunk[1]
        ds.append(baseD)

protoDeBTs = [1]
for idxR,r in enumerate(rs):
    for idxD,d in enumerate(ds):
        a = makeDeBT(r,d,5,3)
        if not a.any() == 3:
            print(a)
            print(f'{idxR} {idxD}\n')
            # for protoDeBT in protoDeBTs:
            #     if not np.array_equal(a,protoDeBT):
            #         protoDeBTs.append(a)
            #         protoDeBTs.append(f'{idxR} {idxD}\n')

# for protoDeBT in protoDeBTs:
#     print(protoDeBT)

print(rs[95])
print(ds[9])