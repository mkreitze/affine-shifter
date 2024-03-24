import numpy as np
import itertools as it
import matplotlib.pyplot as plt

# A quick check to see if the mod operation messes with RDW = DRW 



# sets up the bases
rBase = np.zeros((4,4),dtype=int) 
rBase[0,2] = 1
rBase[1,3] = 1
dBase = np.zeros((4,4),dtype=int) 
dBase[0,1] = 1
dBase[2,3] = 1
rs = [];ds = [];ws=[]

windows = it.product([0,1],repeat = 4)
for win in windows:
    newWin = np.array(win,dtype = int) # this technically includes the 0 vector but... w.e
    newWin.shape = (len(newWin),1)  
    ws.append(newWin)

matrixPoss = it.product([0,1],repeat = 8) # generates all possible combinations for matricies

for win in matrixPoss: # makes all R/Ds 
    newR = np.copy(rBase)
    newD = np.copy(dBase)
    newR[2] = win[0:4];newR[3] = win[4:8]
    newD[1] = win[0:4];newD[3] = win[4:8]
    rs.append(newR);ds.append(newD)

output = open("2x2testingRDW.txt","w")

output2 = open("2x2testingRD.txt","w")
zeros = np.zeros((4,1),dtype=int)

# Checks: if RDW = DRW and then if RD != DR code outputs case and stops when it finds it 
count = 0 
for r in rs:
  for d in ds:
    for w in ws:
        rdw = (r@(d@w)%2)%2
        drw = (d@(r@w)%2)%2
        rd = (r@d)%2
        dr = (d@r)%2
        # in the future I will parse through the not full rank matricies. rd and dr should also be full rank to achieve all possible windows (right?)
        # I also make sure w is not the 0 vector
        if (np.linalg.det(rd)!=0) and (np.linalg.det(dr)!=0) and (np.linalg.det(r)!=0) and (np.linalg.det(d)!= 0) and (not np.array_equal(w,zeros)):
            if np.array_equal(rdw,drw):
               if not np.array_equal(rd,dr):
                    print("Oh no"); count+=1
                    output.write(f"Case {count} found \n")
                    output.write(f"r \n {str(r)} \n")
                    output.write(f"d \n {str(d)} \n")
                    output.write(f"rd \n {str(rd)} \n")
                    output.write(f"dr \n {str(dr)} \n")
                    output.write(f"w0 \n {str(w)} \n")
                    output.write(f"rdw \n {str(rdw)} \n")
                    output.write(f"drw \n {str(drw)} \n")
print(count)

# # Checks: if RD = DR and then if RDW != DRW code outputs case and stops when it finds it 
# for r in rs:
#   for d in ds:
#     for w in ws:
#         rdw = (r@(d@w)%2)%2
#         drw = (d@(r@w)%2)%2
#         rd = (r@d)%2
#         dr = (d@r)%2
#         # in the future I will parse through the not full rank matricies. rd and dr should also be full rank to achieve all possible windows (right?)
#         if (np.linalg.det(rd)!=0) and (np.linalg.det(dr)!=0) and (np.linalg.det(r)!=0) and (np.linalg.det(d)!= 0):
#             if np.array_equal(rd,dr):
#                 if not np.array_equal(rdw,drw):
#                     print("Oh no")                    
#                     output2.write(f"r \n {str(r)} \n")
#                     output2.write(f"d \n {str(d)} \n")
#                     output2.write(f"rd \n {str(rd)} \n")
#                     output2.write(f"dr \n {str(dr)} \n")
#                     output2.write(f"w0 \n {str(w)} \n")
#                     output2.write(f"rdw \n {str(rdw)} \n")
#                     output2.write(f"drw \n {str(drw)} \n")
#                     quit()