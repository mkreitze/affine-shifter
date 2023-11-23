# generic imports
import jax.numpy as jnp
import jax as j
import matplotlib.pyplot as plt
import numpy as np
import itertools
import math
from jax import lax

SANITY = False  # Runs through case, self validation here

if SANITY:
    import shiftDeBT
 
with j.default_device(j.devices("cpu")[0]): # forces cpu cause cuda is annoying


    def makeKernal(alphabetSize,numOfCells):
        # This makes the kernal for crosses
        kernel = jnp.zeros((2, 2, 1, 1), dtype=jnp.int32)
        # Has: O = 3, I = 2, H = 0, W = 1
        kernel += jnp.array(alphabetSize**(np.arange(numOfCells,dtype=int)).reshape(2,2))[:, :, jnp.newaxis, jnp.newaxis] # this is a very cool way to generate values
        if SANITY:
            print("Kernal for neighbourhood")
            print(kernel[:, :, 0, 0])
        return(kernel)

    def runConv(deBT,kernel):
        out = lax.conv(jnp.transpose(deBT,[0,3,1,2]),    # NCHW image
                    jnp.transpose(kernel,[3,2,0,1]), # OIHW kernel
                    (1, 1),  
                    'SAME') # padding mode
        return(np.array(out)[0,0,:-1,:-1])

    def checkIfTori(RDdeBT,n,m,alphabetSize,numOfCells):
        deBT = jnp.zeros((1,2*n,2*m,1) , dtype=jnp.int32) # Has: N = 0, C = 3, H = 1, W = 2 
        tiled = jnp.tile(RDdeBT,(2,2))
        deBT = deBT.at[0,:,:,0].set(tiled)  
        kernel = makeKernal(alphabetSize,numOfCells)
        conv = runConv(deBT,kernel)
        if SANITY:
            print("Guessed deBT")
            print(RDdeBT)
            print("Tiled")
            print(tiled)
            print("Copied")
            print(deBT[0,:,:,0])
            print("Convoed")
            print(conv)
        isUnique = checkUnique(conv,alphabetSize**numOfCells)

        return(isUnique)
    #   isUnique = checkUnique(A**n,conv)
    #   return(isUnique)

    def checkUnique(conved,kN):
        allNums = jnp.arange(0,kN)
        n = int(math.floor((conved.shape[0]+1)/2))
        m = int(math.floor((conved.shape[1]+1)/2))
        convedTori = conved[0:n,0:m].flatten()
        sorted = np.sort(convedTori)
        isDeBT = np.array_equal(sorted,allNums)
        if SANITY:
            print(n,m)
            print("Flattened")
            print(convedTori)
            print("Sorted")
            print(sorted)
            print("Is deBT?")
            print(isDeBT)
        return(isDeBT)



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
        alphabetSize = 2
        numOfCells = 4

        RDdeBT = shiftDeBT.makeDeBT(RGood,DGood,4,4,shiftDeBT.INITWINDOW,shiftDeBT.ALPHABETSIZE)
        n = 4
        m = 4
        print(checkIfTori(RDdeBT,4,4,alphabetSize,numOfCells))