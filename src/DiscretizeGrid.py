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

  def generateTestEmmisivity(self,regionSize, solverParamsFile):
    blk=50
    blk=5
    gds = GDSLoader(regionSize, solverParamsFile)
    gds1 = np.vstack((0.89*np.ones((blk,blk)), 
                      0.78*np.ones((blk,blk)),
                      0.70*np.ones((blk,blk)),
                      0.80*np.ones((blk,blk)),
                      ))
    gds2 = np.vstack((0.75*np.ones((blk,blk)),
                      0.82*np.ones((blk,blk)),
                      0.70*np.ones((blk,blk)),
                      0.85*np.ones((blk,blk)),
                      ))
    gds3 = np.hstack((gds1,gds2))
    
    #gds3 = np.vstack((0.89*np.ones((blk,blk)),
    #                  0.70*np.ones((blk,blk))))

    #gds3 = gds1                      

    gds_k_sio2 = gds3*gds.sio2_k
    gds_k_si = (1-gds3)*gds.si_k 

    gds.e = gds3
    gds.k_si = gds_k_si
    gds.k_sio2 = gds_k_sio2
    gds.k = gds.k_si + gds.k_sio2 
    self.regionSize = regionSize*1e-6

    self.gds = gds
    return gds

  def getEmmissivityMap(self,npzFiles,regionSize, solverParamsFile, half_mode):
    st = time()
    #gds = GDSLoader(regionSize, solverParamsFile)
    self.logger.log(25,"Creating emissivity map from GDS NPZ")
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

  def visualizeEmmissivity(self,gds, regionSize):
    plt.figure()
    plt.imshow(gds.e.T, cmap='jet')
    plt.title("Emissivity map at %d um resolution"%regionSize)
    plt.xlabel('Width (x10 um)')
    plt.ylabel('Length (x10 um)')
    self.logger.info("Average emissivity: %5.3f"% np.mean(gds.e))
    plt.colorbar()

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

  def createNodes(self, gds):
    disMap,max_dis = self.createDiscretization(gds.e)
    
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
    plt.figure()
    plt.imshow(disMap.T, cmap='jet')
    plt.colorbar()
    #for (i,j),label in np.ndenumerate(node_nums):
    #  plt.text(i,j,"%d"%label,ha='center',va='center')

    
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
        

#@jit(nopython=True)
#def calQT(T, e, dz, sigma, lampTemp):
#  #q_t = self.sigma * self.e[...,np.newaxis] * area * (self.lampThermalProfile(t)**4 - T**4)
#  #q_t/(area*dz3d)
#
#  #q_t = sigma * e * (lampTemp**4 - T**4) /(2*dz)
#
#  #node_num = 0
#  q_t = np.zeros_like(T)
#  for n in range(q_t.size):
#    q_t[n] = sigma * e[n] * (lampTemp**4 - T[n]**4 ) / (2*dz[n])
#  #q_t = np.zeros_like(T)
#  #for z, nodes in enumerate(self.grid.nodes):
#  #  for node in nodes:
#  #    node.setTemp(T[node_num])
#  #    q_t[node_num] = ( self.sigma * 
#  #                      node.e * 
#  #                      (self.lampThermalProfile(t)**4 -node.T**4) / 
#  #                      (2*node.dz)
#  #                    )
#  #    node_num += 1
#  return q_t


class discreteSolver():
  def __init__(self, discrete_grid, solverParams, ksiScaling):
    self.grid = discrete_grid
    self.build(solverParams)
    self.ksiScaling = ksiScaling

  def defineParameters(self, solverParamsFile):
    self.solverParamsFile = solverParamsFile
    with solverParamsFile.open('r') as f:
      solver_params = json.load(f)
    self.thickness =solver_params['thickness'] 
    self.dz = np.array(solver_params['dz'])
    self.die_size = solver_params['die_size'] 
    self.temp_lamp_on = solver_params['temp_lamp_on'] 
    self.temp_lamp_off = solver_params['temp_lamp_off'] 
    self.density = solver_params['density'] 
    self.cp = solver_params['cp'] 
    self.si_k = solver_params['si_k'] 
    self.sio2_k = solver_params['sio2_k'] 
    self.sigma = solver_params['sigma'] 
    self.T_bound = solver_params['T_bound'] 
    self.k_bound = solver_params['k_bound'] 
    self.dz_bound = solver_params['dz_bound'] 
    self.r_sd = solver_params['r_sd'] 
    self.r_ti = solver_params['r_ti'] 
    self.r_gate = solver_params['r_gate']
  
  def build(self, solverParams):
    self.defineParameters(solverParams)


  def runSolver(self,t_max, time_step, pw_lamp):
    np.savez_compressed('./results/discrete_tests/e_reg_%d_pul_%d.npz'%(
                        self.grid.regionSize, pw_lamp*1e3), T=self.grid.gds.e)
    self.t_lamp = pw_lamp
    self.t_step = time_step  # 0.1
    self.t_eval = np.arange(0,t_max,time_step)
    T_init = self.T_bound*np.ones((self.grid.num_nodes,))
    t1 = time()
    self.pbar = tqdm(total=t_max,position=0,leave=True,
      bar_format = "{desc}: {percentage:.1f}%|{bar}| {n:.1e}/{total:.1e} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    self.t_last = 0
    self.tot_time = 0
    self.qt_time = 0
    self.kt_time = 0
    self.calc_time = 0
    self.kt_count=0
    print("num nodes: %d"%self.grid.num_nodes)
    print("z size: %d"%len(self.dz))
    sol = solve_ivp(self.differential, 
                    [0, t_max], 
                    T_init, 
                    t_eval=self.t_eval, 
                    #vectorized=False,
                    vectorized=True,
                    method = 'BDF',
                    rtol=0.5e-2,
                    atol=0.5e-4) 
                    #method = 'LSODA')
    #sol = odeint(self.differential_ode, 
    #                #[0, t_max], 
    #                T_init, 
    #                self.t_eval,
    #                tfirst=True)#, 
    #                #vectorized=False,
    #                #vectorized=True,
    #                #method = 'Radau') 
    #                #method = 'LSODA')
    temp = sol.y.T
    
    print("Total times:")
    print("QT: %5.2f"%self.qt_time)
    print("KT: %5.2f"%self.kt_time)
    print("Calc:%5.2f"%self.calc_time)

    node_num = 0
    temp_map = np.zeros((temp.shape[0], self.grid.nx, self.grid.ny))
    for node in self.grid.nodes[0]:
      llx, urx, lly, ury = node.llx, node.urx, node.lly, node.ury
      temp_map[:,llx:urx,lly:ury] = temp[:, node_num].reshape(-1,1,1)
      node_num += 1 

    temp_range = np.mean(temp_map,axis=(1,2))
    plt.figure()
    plt.plot(self.t_eval,temp_range)
    self.pbar.close()
    plt.figure()
    plt.imshow(temp_map[9].T,cmap='jet')
    plt.colorbar()
    np.savez_compressed('./results/discrete_tests/T_reg_%d_pul_%d.npz'%(
                        self.grid.regionSize, pw_lamp*1e3), T=temp_map)
    #self.saveData()
    #k_maps = np.zeros((temp.shape[0], self.grid.nx, self.grid.ny))
    #k_si_maps = np.zeros((temp.shape[0], self.grid.nx, self.grid.ny))
    #k_sio2_maps = np.zeros((temp.shape[0], self.grid.nx, self.grid.ny))
    #for n in range(temp_map.shape[0]):
    #  node_num = 0
    #  for node in self.grid.nodes[0]:
    #    llx, urx, lly, ury = node.llx, node.urx, node.lly, node.ury
    #    node.setTemp(temp[n, node_num])
    #    k_maps[n,llx:urx,lly:ury] = node.conductivity(self.ksiScaling)
    #    k_si_maps[n,llx:urx,lly:ury] = node.siConductivity(self.ksiScaling)
    #    k_sio2_maps[n,llx:urx,lly:ury] = node.k_sio2
    #    node_num += 1 
    #max_val = np.max(k_maps)
    #min_val = np.min(k_maps)
    #for n,k_map in enumerate(k_maps):
    #  ind = np.unravel_index(np.argmax(k_map, axis=None), k_map.shape)
    #  print(temp_map[n][ind] ,k_map[ind])
    #  ind = np.unravel_index(np.argmax(k_si_maps[n], axis=None), k_map.shape)
    #  print(temp_map[n][ind] ,k_si_maps[n][ind])
    #  ind = np.unravel_index(np.argmax(k_sio2_maps[n], axis=None), k_map.shape)
    #  print(temp_map[n][ind] ,k_sio2_maps[n][ind])
    #  
      #plt.figure()
      #plt.imshow(k_map, cmap='jet', vmin = min_val, vmax= max_val)
      #plt.colorbar()
      #plt.title('at %f'%self.t_eval[n])
      

  def lampThermalProfile(self, t):
    if t <= self.t_lamp:
      return self.temp_lamp_on
    else:
      return self.temp_lamp_off



  def calKT(self, n_points):
    node_num = 0
    if n_points==0:
      dkt_dx2 = np.zeros((self.grid.num_nodes,1))
    else:
      dkt_dx2 = np.zeros((self.grid.num_nodes,n_points))
      
    self.kt_count+=1
    for z, nodes in enumerate(self.grid.nodes):
      for node in nodes:
        for direction, direction_nodes in enumerate([node.east, node.west, node.north, node.south]): 
          num_res = len(direction_nodes)
          for n_node in direction_nodes: 
            if direction<2:
              d1 = node.dx 
              d2 = n_node.dx
            else:
              d1 = node.dy 
              d2 = n_node.dy
            dd = d1 + d2
            k_res = dd/(d1/node.conductivity(self.ksiScaling) + d2/n_node.conductivity(self.ksiScaling))
            dt = n_node.T - node.T
            dkt_dx2[node_num] += k_res*dt/(dd**2)

        for direction_nodes in [node.top, node.bottom]:
          if len(direction_nodes) == 0:
            k_res =(node.dz+self.dz_bound)/(
                    node.dz/node.conductivity(self.ksiScaling) + 
                    self.dz_bound/self.k_bound)
            dz = self.dz_bound + node.dz
            dt = self.T_bound - node.T
            dkt_dx2[node_num] += k_res*dt/(dz**2)
          else:
            for n_node in direction_nodes: 
              k_res = (node.dz+n_node.dz)/( node.dz/node.conductivity(self.ksiScaling) +
                        n_node.dz/n_node.conductivity(self.ksiScaling))
              dt = n_node.T - node.T
              dz = node.dz + n_node.dz
              dkt_dx2[node_num] += k_res*dt/(dz**2)
        node_num += 1
    return dkt_dx2

  def differential_ode(self,t,T):
    stot = time()
    tdiff = t-self.t_last
    if tdiff>0:
      self.pbar.update(tdiff)
      self.t_last = t
    self.pbar.set_postfix({
          'tot': '%5.2f'%self.tot_time,
          'qt': '%5.2f'%self.qt_time,
          'kt': '%5.2f'%self.kt_time,
          'n':'%d'%self.kt_count,
          'calc': '%5.2f'%self.calc_time
      })
    
    #print("T",T.shape, len(T.shape))
    #print(type(T))
    if len(T.shape) == 1:
      n_points = 0
      T = T.reshape(-1,1)
      #print("reshaped", T.shape)
    else:
      n_points = T.shape[1]
      #print("not reshaped", T.shape)
    #print("T",T.shape)
    sqt = time()
    node_num = 0
    q_t = np.zeros_like(T)
    for z, nodes in enumerate(self.grid.nodes):
      for node in nodes:
        node.setTemp(T[node_num,:])
        q_t[node_num] = ( self.sigma * 
                          node.e *
                          (self.lampThermalProfile(t)**4 -node.T**4) / 
                          (2*node.dz) 
                        )
        node_num += 1
    #print("q_t",q_t.shape)
    skt = time()
    self.qt_time += (skt - sqt)
    dkt_dx2 = self.calKT(n_points) 
    #print("dkt_dx2",dkt_dx2.shape)
    scalc = time()
    self.kt_time += (scalc - skt)
    dTdt_ = (1/(self.cp*self.density)) * ( dkt_dx2  + q_t)
    self.calc_time += (time()-scalc)
    #print("dTdt",dTdt_.shape)
    if n_points==0:
      dTdt_ = dTdt_.reshape(-1)
    else:
      dTdt_ = dTdt_.reshape(-1,n_points)
    #print("dTdt",dTdt_.shape)
    #print(np.max(dTdt_))
    #print(np.mean(dTdt_))
    self.tot_time += time()-stot
    return dTdt_
  
  def differential(self,t,T):
    stot = time()
    tdiff = t-self.t_last
    if tdiff>0:
      self.pbar.update(tdiff)
      self.t_last = t
    self.pbar.set_postfix({
          'tot': '%5.2f'%self.tot_time,
          'qt': '%5.2f'%self.qt_time,
          'kt': '%5.2f'%self.kt_time,
          'n':'%d'%self.kt_count,
          'calc': '%5.2f'%self.calc_time
      })
    
    #print(T.shape, len(T.shape))
    #print(type(T))
    if len(T.shape) == 1:
      n_points = 0
      T = T.reshape(-1,1)
      #print("reshaped", T.shape)
    else:
      n_points = T.shape[1]
      #print("not reshaped", T.shape)
    sqt = time()
    node_num = 0
    q_t = np.zeros_like(T)
    for z, nodes in enumerate(self.grid.nodes):
      for node in nodes:
        node.setTemp(T[node_num,:])
        q_t[node_num] = ( self.sigma * 
                          node.e *
                          (self.lampThermalProfile(t)**4 -node.T**4) / 
                          (2*node.dz) 
                        )
        node_num += 1
    skt = time()
    self.qt_time += (skt - sqt)
    dkt_dx2 = self.calKT(n_points) 
    scalc = time()
    self.kt_time += (scalc - skt)
    dTdt_ = (1/(self.cp*self.density)) * ( dkt_dx2  + q_t)
    self.calc_time += (time()-scalc)
    if n_points==0:
      dTdt_ = dTdt_.reshape(-1)
    else:
      dTdt_ = dTdt_.reshape(-1,n_points)
    #print(np.max(dTdt_))
    #print(np.mean(dTdt_))
    self.tot_time += time()-stot
    return dTdt_

def plot_maps(grid):
  node_num = 0
  k_map = np.zeros((grid.nx, grid.ny))
  e_map = np.zeros((grid.nx, grid.ny))
  for node in grid.nodes[0]:
    llx, urx, lly, ury = node.llx, node.urx, node.lly, node.ury
    k_map[llx:urx,lly:ury] = node.conductivity() 
    e_map[llx:urx,lly:ury] = node.e
    node_num += 1 
  plt.figure()
  plt.imshow(k_map.T,cmap='jet')
  plt.colorbar()
  plt.title('k_map')
  plt.figure()
  plt.imshow(e_map.T,cmap='jet')
  plt.colorbar()
  plt.title('e_map')
  plt.show()
  exit()
    

def main():
  dsg_h = DiscretizeGrid()
  #use_testcase = True
  use_testcase = False
  ksiScaling = True
  half_mode = True

  regionSize =40#10
  solverParamsFile = Path('config/SolverParams.json')
  if use_testcase:
    gds = dsg_h.generateTestEmmisivity(regionSize, solverParamsFile)

  else:
    npzFiles = []
    npzFiles.append( Path('results/test/TCAD_RTP_A.npz'))
    npzFiles.append( Path('results/test_RTP_B/TCAD_RTP_B.npz'))
    gds = dsg_h.getEmmissivityMap(npzFiles,regionSize, solverParamsFile, half_mode)
  
  print("main")
  print(gds.e.shape)
  dsg_h.visualizeEmmissivity(gds,regionSize)
  dsg_h.createNodes(gds) 
  #plt.show()
  #plot_maps(dsg_h)
  simulator = discreteSolver(dsg_h, solverParamsFile, ksiScaling)
  t_max = 12e-3 
  t_step = 10e-4 
  pw_lamp = 10e-3 
  simulator.runSolver(t_max, t_step, pw_lamp)

if __name__ == "__main__":
  main()
  plt.show()
   
