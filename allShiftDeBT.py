

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import itertools as it
import math as m

# User inputs go here:

ALPHABET = [0,1] # enumerate alphabet, this is needed for determinant conditions

WINDOW = np.array(
    [[1,2],
     [3,4]]) # The window for the desired deBT

INITWINDOW = np.array([0,0,0,0,1],dtype = int)
INITWINDOW.shape = (5,1) # Desired initial window in the top left spot

MAIN = True # If you are running this file, set to True


WINDOWSIZE = len(np.nonzero(WINDOW[0]))
ALPHABETSIZE = len(ALPHABET)
OUTPUT = f"deBT for |W| = {WINDOWSIZE} and |A| ={len(ALPHABET)}.txt"
MAXSIZE = WINDOWSIZE**ALPHABETSIZE

# INPUT: numpy array {matrix}, max tori dimension {order}, finite field size {alphabetSize} 
# Generates powers of {matrix} until the identity is generated.
# OUTPUT: integer {cyclicPower}, all matrix powers from 1 until {i}

## NOTES: Because of formulation, identity matrix will always be last.
## NOTES: Broadcasting all powers and then a check _may_ be faster.
## NOTES: If the identity is not generated before the max tori dimension is reached, spits out an empty array and {cyclicPower = 0}
def genPowers(matrix,maxSize,alphabetSize):
    pows = []    
    I = np.eye(matrix.shape[0])
    for i in range(1,maxSize):
        temp = np.linalg.matrix_power(matrix, i)%alphabetSize
        pows.append(temp)
        if np.array_equal(temp,I):
            return(i,np.array(pows))
    return(0,[])

# INPUT: numpy array {R}, numpy array {D}, finite field size {alphabetSize}
# Checks if two finite field arrays are commutable
# OUTUT: boolean {commuteable}

def isCommutable(R,D,alphabetSize):
    O = (R @ D) % alphabetSize
    P = (D @ R) % alphabetSize
    return(np.array_equal(O,P))

# INPUT: integer {num}
# OUTPUT: all factors of {num}

# NOTE: Taken from user Sublow at https://stackoverflow.com/questions/47064885/list-all-factors-of-number
def factorize(num):
    return [n for n in range(1, num + 1) if num % n == 0]


def searchValids():
    return(0)

def generateAllR():
    return(0)

def generateAllD():
    return(0)


# if MAIN:
#     # Make all generic R and Ds
#     unFilteredR = genRShifters()
#     unFilteredD = genDShifters()
#     # Sort by powers
#     rPows = genPowers() 
#     dPows = genPowers()
#     # Go through each valid power combo
#     for Rs in rPows:
#         for Ds in dPows:
#             if isCommutable(Rs[0],Ds[0]):


