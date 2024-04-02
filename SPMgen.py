import numpy as np
import itertools as it
import matplotlib.pyplot as plt


# I made a colab file to test specific parts 
# https://colab.research.google.com/drive/1ZKirfZeCMSzYT93zUFyy3s2DRt9z-LoN#scrollTo=a2MNHbWPdHJz&uniqifier=2 

# INPUTS 
WINDOW = np.array(
    [[1,3],
     [2,4]])
WINDOWSIZE = len(np.nonzero(WINDOW)[0])
ALPHABET = [0,1]
ALPHABETSIZE = len(ALPHABET)
INITWINDOW = np.array([0,0,0,0],dtype = int)

INITWINDOW.shape = (len(INITWINDOW),1) # Needed incase np array initalization screws up 
SANITY = False # Prints out stats to make sure things are working/for demo


def makeFrontPair(shifted,mask,window):
  relShift = np.multiply(shifted,mask)
  nonZeros = np.nonzero(relShift)
  front1 = relShift[nonZeros]-1
  front2 = window[nonZeros]-1
  return(front1,front2)

def generateFronts(window):
  mask = np.copy(window)
  mask[mask != 0] = 1
  down = np.eye(np.shape(window)[0],dtype = int)
  down = np.roll(down,1,axis=0)
  down[:,-1] = 0
  frontPairUD = makeFrontPair(down@window,mask,window)
  frontPairLR = makeFrontPair(window@down.transpose(),mask,window)
  frontPairs = [frontPairLR,frontPairUD]
  return(frontPairs)

def generateGeneric(windowSize,frontPair):
  generic = np.zeros((windowSize,windowSize))
  generic[frontPair[0],frontPair[1]] = 1
  return(generic)


fronts = generateFronts(WINDOW)
print(f" lFront = {fronts[0][0]} \n rFront = {fronts[0][1]} \n uFront = {fronts[1][0]} \n dFront = {fronts[1][1]}")


genericR = generateGeneric(WINDOWSIZE,fronts[0])
genericD = generateGeneric(WINDOWSIZE,fronts[1])
print(genericR)
print(genericD)

