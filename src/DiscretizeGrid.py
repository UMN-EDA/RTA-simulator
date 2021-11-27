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
import json
from tqdm import tqdm, trange
from numba import jit, njit
from numba.experimental import jitclass

from scipy.integrate import solve_ivp
from scipy.integrate import odeint

class node():
  def __init__(self,num,llx,urx,lly,ury, 
                    dx,dy,dz, 
                    k_si, k_sio2, e, # characteristics of the cell
                    si_k, sio2_k # conductivity values of si and sio2
                    ):
    self.logger = logging.getLogger('TAZ.NDE')
    self.num = num
    self.llx = llx
    self.urx = urx
    self.lly = lly
    self.ury = ury
    self.dx = dx*abs(urx-llx)/2
    self.dy = dy*abs(ury-lly)/2
    self.a =  abs(urx-llx)* abs(ury-lly)
    self.dz = dz/2
    self.e = e
    self.T = 0
    self.k_si = k_si
    self.k_sio2 = k_sio2
    self.si_k = si_k
    self.sio2_k = sio2_k
    self.east = []
    self.west = []
    self.north = []
    self.south = []
    self.top = []
    self.bottom = []
    self.nums_east = []
    self.nums_west = []
    self.nums_north = []
    self.nums_south = []
    self.nums_top = []
    self.nums_bottom = []
  
  @property
  def area(self):
    return 4*self.dx*self.dy 

  def addEast(self, node):
    if not node.num in self.nums_east:
      self.east.append(node)
      self.nums_east.append(node.num)

  def addWest(self, node):
    if not node.num in self.nums_west:
      self.west.append(node)
      self.nums_west.append(node.num)

  def addNorth(self, node):
    if not node.num in self.nums_north:
      self.north.append(node)
      self.nums_north.append(node.num)

  def addSouth(self, node):
    if not node.num in self.nums_south:
      self.south.append(node)
      self.nums_south.append(node.num)

  def addTop(self, node):
    if not node.num in self.nums_top:
      self.top.append(node)
      self.nums_top.append(node.num)

  def addBottom(self, node):
    if not node.num in self.nums_bottom:
      self.bottom.append(node)
      self.nums_bottom.append(node.num)

  def setTemp(self, T):
    self.T = T

  def thermalConductivity(self, T, ksiScaling):
    if ksiScaling:
      #Si thermal confuctivity profile in W/cmK
      p1 = -1.225
      p2 =  957.4
      p3 = -3356
      q1 = -45.22
      q2 = 1183
      fxn = (p1*(T**2) + p2*T + p3) / ((T**2) + q1*T + q2)
      fxn = fxn*100 # convert to W/mK
      # capping the lower conductivity to conductivity of SiO2 to prevent
      # inaccuracies in extrapolation
      fxn = np.maximum(fxn,self.sio2_k)
    else:
      fxn=self.si_k
    return fxn  
  def siConductivity(self, ksiScaling):
    fxn = self.thermalConductivity(self.T,ksiScaling)
    k_si = (fxn/self.si_k) * self.k_si
    return k_si

  def conductivity(self, ksiScaling):
    fxn = self.thermalConductivity(self.T,ksiScaling)
    self.k = self.k_sio2 + (fxn/self.si_k) * self.k_si
    #self.k = self.k_si + self.k_sio2
    return self.k
    
  def print(self):
    self.logger.debug("Node @: [ ll %d %d , ur %d %d ]"%(llx,lly,urx,ury))
    self.logger.debug("Temperature: %5.2f"%self.T)
    self.logger.debug("Conductivity: %5.2f"%self.conductivity())

class DiscretizeGrid():
  def __init__(self):
    self.createLogger()
    self.nodes = []
  
  @property 
  def num_nodes(self):
    num_nodes = 0
    for z_nodes in self.nodes:
      num_nodes += len(z_nodes)
    return num_nodes

  def createLogger(self):
    self.logger = logging.getLogger('TAZ.DSG')

  def getEmmissivityMap(self,npzFiles,regionSize, solverParamsFile, half_mode):
    st = time()
    #gds = GDSLoader(regionSize, solverParamsFile)
    #self.logger.log(25,"Creating emissivity map from GDS NPZ")
    for n, npzFile in enumerate(npzFiles):
      gds_n = GDSLoader(regionSize, solverParamsFile)
      gds_n.createEmmissivityMatrix(npzFile)
      dimx, dimy =  gds_n.e.shape
      if half_mode:
        dimx = int(dimx/2)
      if n == 0:
        gds = gds_n
        gds.e = gds_n.e[0:dimx, :]
        gds.k_si =  gds_n.k_si[0:dimx, :]
        gds.k_sio2 =  gds_n.k_sio2[0:dimx, :]
      else:
        #print(gds.e.shape,gds_n.e)
        gds.e = np.vstack((gds.e, gds_n.e[0:dimx, :]))
        gds.k_si = np.vstack((gds.k_si, gds_n.k_si[0:dimx, :]))
        gds.k_sio2 = np.vstack((gds.k_sio2, gds_n.k_sio2[0:dimx, :]))
    gds.k = gds.k_si + gds.k_sio2
    
    self.regionSize = regionSize*1e-6
    self.gds = gds
    return gds

  def createDiscretization(self, emmMap):
    emm_map = np.array(emmMap)

    merge = np.zeros_like(emm_map)
    merge_size = np.ones_like(emm_map) 
    merge[ 0, :] = 1
    merge[-1, :] = 1
    merge[ :,-1] = 1
    merge[ :, 0] = 1

    max_val = 0
    for x in range(1,emm_map.shape[0]-1):
      for y in range(1,emm_map.shape[1]-1):
        if np.all(abs(emm_map[x-1:x+2,y-1:y+2] - emm_map[x,y])<5e-3):
          merge[x,y] = 1
          max_val = 1
    merge1 = np.array(merge)
    for x in range(2,emm_map.shape[0]-4,2):
      for y in range(2,emm_map.shape[1]-4,2):
        if np.all(merge[x-2:x+4,y-2:y+4] == 1):
          merge1[x:x+2,y:y+2] = 2
          max_val = 2
    merge2 = np.array(merge1)
    for x in range(4,emm_map.shape[0]-8,4):
      for y in range(4,emm_map.shape[1]-8,4):
        if np.all(merge1[x-4:x+8,y-4:y+8] == 2):
          merge2[x:x+4,y:y+4] = 3
          max_val = 3
    merge3 = np.array(merge2)
    for x in range(8,emm_map.shape[0]-16,8):
      for y in range(8,emm_map.shape[1]-16,8):
        if np.all(merge2[x-8:x+16,y-8:y+16] == 3):
          merge3[x:x+8,y:y+8] = 4
          max_val = 4
    merge4 = np.array(merge3)
    for x in range(16,emm_map.shape[0]-32,16):
      for y in range(16,emm_map.shape[1]-32,16):
        if np.all(merge3[x-16:x+32,y-16:y+32] == 4):
          merge4[x:x+16,y:y+16] = 5
          max_val = 5
    merge5 = np.array(merge4)
    for x in range(32,emm_map.shape[0]-64,32):
      for y in range(32,emm_map.shape[1]-64,32):
        if np.all(merge4[x-32:x+64,y-32:y+64] == 5):
          merge5[x:x+32,y:y+32] = 6
          max_val = 6
    return merge5, max_val

  def createNodes(self, gds, enableDiscretization):
    if enableDiscretization:
      disMap,max_dis = self.createDiscretization(gds.e)
    else:
      disMap,max_dis = np.zeros_like(gds.e), 0
      
    nz = gds.dz.shape[0]
    # to force no dicretization or maximum discretization
    #max_dis = 0

    self.nx = gds.e.shape[0]
    self.ny = gds.e.shape[1]
    self.nz = nz
    node_nums = -1*np.ones((nz,)+disMap.shape,dtype=np.int)
    node_num = np.zeros((nz,),dtype=np.int)
    for z in range(nz):
      self.nodes.append([])
    for n in range(max_dis, -1, -1):
      sz = 2**n
      for x in range(0, disMap.shape[0]-sz+1,1):
        for y in range(0, disMap.shape[1]-sz+1,1):
          dis_region = disMap[x:x+sz,y:y+sz]
          node_region = node_nums[0,x:x+sz,y:y+sz]
          if np.all(dis_region >= n) and np.all(node_region == -1): 
            llx, urx = x,x+sz
            lly, ury = y,y+sz
            k_si = np.mean(gds.k_si[llx:urx,lly:ury])
            k_sio2 = np.mean(gds.k_sio2[llx:urx,lly:ury])
            for z in range(nz):
              node_nums[z,x:x+sz,y:y+sz] = node_num[z]
              if z == 0:
                e = np.mean(gds.e[llx:urx,lly:ury])
              else:
                e = 0
              node_h = node(node_num[z],llx,urx,lly,ury, 
                            self.regionSize, self.regionSize, gds.dz[z], 
                            k_si, k_sio2, e, gds.si_k, gds.sio2_k)
              self.nodes[z].append(node_h)
              node_num[z] += 1
    
    for x in range(0, node_nums.shape[1], 1):
      for y in range(0, node_nums.shape[2], 1):
        for z in range(0, node_nums.shape[0], 1):
          n = node_nums[z,x,y]
          n_h = self.nodes[z][n]
          if x!=node_nums.shape[1]-1:
            ne = node_nums[z,x+1,y]
          else:
            ne = node_nums[z,0,y]
          if y!=node_nums.shape[2]-1:
            ns = node_nums[z,x,y+1]
          else:
            ns = node_nums[z,x,0]
          
          if z!=node_nums.shape[0]-1:
            nb_h = self.nodes[z+1][n]
            n_h.addBottom(nb_h)
            nb_h.addTop(n_h)
          
          if ne != n:
            ne_h = self.nodes[z][ne]
            n_h.addEast(ne_h)
            ne_h.addWest(n_h)

          if ns != n:
            ns_h = self.nodes[z][ns]
            n_h.addSouth(ns_h)
            ns_h.addNorth(n_h)
    self.e = np.zeros((self.num_nodes,))     
    self.dz = np.zeros((self.num_nodes,))     
    node_num = 0
    for z, nodes in enumerate(self.nodes):
      for node_h in nodes:
        self.e[node_num] = node_h.e
        self.dz[node_num] = node_h.dz
        node_num+=1

