"""
BSD 3-Clause License
Copyright (c) 2021, The Regents of the University of Minnesota
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
# Author: VidyaChhabria

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
from GDSLoader import GDSLoader
from time import time
from pathlib import Path

class DiscretizeGrid():
  def __init__(self):
    self.createLogger()

  def createLogger(self):
    self.logger = logging.getLogger('TAZ.DSG')

  

  def visualizeEmmissivity(self,npzFile,regionSize, solverParamsFile):
    gds1 = np.vstack((0.89*np.ones((100,100)), 0.78*np.ones((100,100)), 0.70*np.ones((100,100))))
    gds2 = np.vstack((0.75*np.ones((100,100)), 0.82*np.ones((100,100)), 0.70*np.ones((100,100))))
    gds3 = np.hstack((gds1,gds2))
    st = time()
    gds = GDSLoader(regionSize, solverParamsFile)
    self.logger.log(25,"Creating emissivity map from GDS NPZ")
    gds.createEmmissivityMatrix(npzFile)
    plt.figure()
    #plt.imshow(gds, cmap='jet')
    plt.imshow(gds.e.T, cmap='jet')
    plt.title("Emissivity map at %d um resolution"%regionSize)
    plt.xlabel('Width (x10 um)')
    plt.ylabel('Length (x10 um)')
    #self.logger.info("Average emissivity: %5.3f"% np.mean(gds.e))
    plt.colorbar()
    self.logger.log(25,"Created emissivity map in %5.2fs"%(time()-st))
    return gds.e
    #return gds3

def main():
  dsg_h = DiscretizeGrid()
  solverParamsFile = Path('config/SolverParams.json')
  npzFile = Path('results/test/TCAD_RTP_A.npz')
  regionSize = 10
  gds = dsg_h.visualizeEmmissivity(npzFile,regionSize, solverParamsFile)
  #emm_map = np.array(gds.e)
  emm_map = np.array(gds)

  merge = np.zeros_like(emm_map)
  merge_size = np.ones_like(emm_map) 
  merge[ 0, :] = 1
  merge[-1, :] = 1
  merge[ :,-1] = 1
  merge[ :, 0] = 1

  for x in range(1,emm_map.shape[0]-1):
    for y in range(1,emm_map.shape[1]-1):
      if np.all(abs(emm_map[x-1:x+2,y-1:y+2] - emm_map[x,y])<2e-3):
        merge[x,y] = 1
  merge1 = np.array(merge)
  for x in range(2,emm_map.shape[0]-4,2):
    for y in range(2,emm_map.shape[1]-4,2):
      if np.all(merge[x-2:x+4,y-2:y+4] == 1):
        merge1[x:x+2,y:y+2] = 2
  merge2 = np.array(merge1)
  for x in range(4,emm_map.shape[0]-8,4):
    for y in range(4,emm_map.shape[1]-8,4):
      if np.all(merge1[x-4:x+8,y-4:y+8] == 2):
        merge2[x:x+4,y:y+4] = 3
  merge3 = np.array(merge2)
  for x in range(8,emm_map.shape[0]-16,8):
    for y in range(8,emm_map.shape[1]-16,8):
      if np.all(merge2[x-8:x+16,y-8:y+16] == 3):
        merge3[x:x+8,y:y+8] = 4
  merge4 = np.array(merge3)
  for x in range(16,emm_map.shape[0]-32,16):
    for y in range(16,emm_map.shape[1]-32,16):
      if np.all(merge3[x-16:x+32,y-16:y+32] == 4):
        merge4[x:x+16,y:y+16] = 5
  merge5 = np.array(merge4)
  for x in range(32,emm_map.shape[0]-64,32):
    for y in range(32,emm_map.shape[1]-64,32):
      if np.all(merge4[x-32:x+64,y-32:y+64] == 5):
        merge5[x:x+32,y:y+32] = 6
   

  plt.figure()
  plt.imshow(merge5.T, cmap='jet')
  plt.colorbar()
      

      




if __name__ == "__main__":
  main()
  plt.show()
   
