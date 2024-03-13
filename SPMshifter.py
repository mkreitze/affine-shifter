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
SANITY = True


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
    return(deBT)

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
        baseR[2,0] = det[0,0]
        baseR[2,1] = det[0,1]
        baseR[3,0] = det[1,0]
        baseR[3,1] = det[1,1]
        baseR[2,2] = freeChunk[0,0]
        baseR[2,3] = freeChunk[0,1]
        baseR[3,2] = freeChunk[1,0]
        baseR[3,3] = freeChunk[1,1]        
        rs.append(baseR)
    


ds = []
detRows = [1,2];detCols = [0,2]
freeRows = [1,3];freeCols = [2,3];freeShape = (2,2)
# makes Ds 
# only applies det condition
for det in dets:
    baseD = initDShifter()
    for frees in allFrees:
        freeChunk = np.array(frees,dtype = int)
        freeChunk.shape = freeShape
        baseD[2,0] = det[0,0]
        baseD[3,0] = det[0,1]
        baseD[1,2] = det[1,0]
        baseD[3,2] = det[1,1]
        baseD[1,1] = freeChunk[0,0]
        baseD[1,3] = freeChunk[0,1]
        baseD[3,1] = freeChunk[1,0]
        baseD[3,3] = freeChunk[1,1]        
        ds.append(baseD)

protoDeBTs = [makeDeBT(rs[0],ds[0],5,3)]
protoDeBTsWithNotes = []

for idxR,r in enumerate(rs):
    for idxD,d in enumerate(ds):
        a = makeDeBT(r,d,5,3)
        rd = (r@d)%ALPHABETSIZE
        dr = (d@r)%ALPHABETSIZE
        if np.array_equal(rd,dr):
            if SANITY:
                print(f'{a} {idxR} {idxD}\n')
            test = True
            for protoDeBT in protoDeBTs:
                if np.array_equal(a,protoDeBT):
                    test = False 
            if test:
                protoDeBTs.append(a)
                protoDeBTsWithNotes.append(f"{a} R {idxR} D {idxD} \n")
                print(idxR)


# print(f"All unique 'deBTS' found:")
# for deBT in protoDeBTsWithNotes:
#     print(deBT)
# print(len(protoDeBTsWithNotes))
