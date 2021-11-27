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

import numpy as np
from tqdm import tqdm
import logging
from GDSLoader import GDSLoader
from matplotlib import pyplot as plt
from time import time
from scipy.integrate import solve_ivp
import json
from DiscretizeGrid import DiscretizeGrid

class ThermalSolver:
  def __init__(self):
    self.createLogger()
    self.ksiScaling = False
    self.grid = DiscretizeGrid()

  def createLogger(self):
    self.logger = logging.getLogger('TAZ.SOL')

  def enableKsiScaling(self, ksiScaling):
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
    self.nz = self.dz.shape[0]

  def build(self, npzFile, regionSize, outDir, halfMode=False,
            enableDiscretization=False):
    self.region_size = regionSize
    if outDir is not None:
      outDir.mkdir(parents=True, exist_ok=True)
    self.outDir=outDir
    st = time()
    self.logger.log(25,"Creating emissivity map from GDS NPZ")
    self.gds = self.grid.getEmmissivityMap( npzFile, regionSize,
                                    self.solverParamsFile, halfMode)
    self.width = self.gds.e.shape[0]*self.region_size*1e-6
    self.length = self.gds.e.shape[1]*self.region_size*1e-6

    self.grid.createNodes(self.gds, enableDiscretization)                                
    self.logger.log(25,"Created emissivity map in %5.2fs"%(time()-st))

  def buildTest(self, patternNum, regionSize, outDir, enableDiscretization=False):
    self.region_size = regionSize
    st = time()
    if outDir is not None:
      outDir.mkdir(parents=True, exist_ok=True)
    self.outDir=outDir
    self.logger.debug("Enabling pattern %d"%patternNum)
    patternMask = self.testcasePatterns(patternNum-1)
    self.setTestcaseParameters(patternMask, self.region_size, self.solverParamsFile)
    self.width = self.gds.e.shape[0]*self.region_size*1e-6
    self.length = self.gds.e.shape[1]*self.region_size*1e-6
    self.grid.createNodes(self.gds, enableDiscretization)                                
    
  def runSolver(self, t_max, time_step, pw_lamp):
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
    sol = solve_ivp(self.differential, 
                    [0, t_max], 
                    T_init, 
                    t_eval=self.t_eval, 
                    vectorized=True,
                    #method = 'Radau') 
                    method = 'BDF') 
    temp = sol.y.T

    node_num = 0
    temp_map = np.zeros((temp.shape[0], self.grid.nx, self.grid.ny, self.nz))
    for z, nodes in enumerate(self.grid.nodes):
      for node in nodes:
        llx, urx, lly, ury = node.llx, node.urx, node.lly, node.ury
        temp_map[:,llx:urx,lly:ury,z] = temp[:, node_num].reshape(-1,1,1)
        node_num += 1
    self.temp = temp_map

    self.logger.info("Completed solution in %5.2fs"%(time()-t1))
    self.pbar.close()
    self.saveData()

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

  def differential(self,t,T):
    stot = time()
    tdiff = t-self.t_last
    if tdiff>0:
      self.pbar.update(tdiff)
      self.t_last = t
    # Optimzation statistics
    #self.pbar.set_postfix({
    #      'tot': '%5.2f'%self.tot_time,
    #      'qt': '%5.2f'%self.qt_time,
    #      'kt': '%5.2f'%self.kt_time,
    #      'n':'%d'%self.kt_count,
    #      'calc': '%5.2f'%self.calc_time
    #  })
    
    if len(T.shape) == 1:
      n_points = 0
      T = T.reshape(-1,1)
    else:
      n_points = T.shape[1]
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
    self.tot_time += time()-stot
    return dTdt_

  def thermalConductivity(self, T):
    if self.ksiScaling:
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
    
  def lampThermalProfile(self, t):
    if t <= self.t_lamp:
      return self.temp_lamp_on
    else:
      return self.temp_lamp_off
  def saveData(self):
    if self.outDir is not None:
      np.savez_compressed(self.outDir / "temperature_solution.npz",
          temp   = self.temp,
          t_lamp = self.t_lamp,
          width  = self.width,
          length = self.length,
          dz = self.dz,
          t_eval = self.t_eval,
          )

  def tempPlot(self):
    t_plot_point = int(self.t_lamp/self.t_step) 
    fig,ax = plt.subplots()
    T_lw_plot = self.temp[t_plot_point,:,:,0].squeeze()
    len_plot = int(100* self.length/self.width)
    width_plot = 100
    im = ax.imshow(T_lw_plot.T , cmap = 'jet', extent= [0,width_plot, len_plot, 0])

    y_label_list = [int(self.length*x*1e6/100) for x in range(0,101,20)]
    ax.set_yticks([y for y in range(0,int(len_plot+1),int(0.2*len_plot))])
    ax.set_yticklabels(y_label_list)
    x_label_list = [int(self.width*x*1e6/100) for x in range(0,101,20)]
    ax.set_xticks([x for x in range(0,int(width_plot+1),int(0.2*width_plot))])
    ax.set_xticklabels(x_label_list)
    ax.set_ylabel("Length (um)")
    ax.set_xlabel("Width (um)")
    fig.colorbar(im)
    plt.show()
    plt.close(fig)

  def testcasePatterns(self, num):
    width = 4e-3
    length = 4e-3
    nx = int(1e6*width/self.region_size +1e-6)
    ny = int(1e6*length/self.region_size +1e-6)

    mask1 = np.zeros((nx,ny))
    mask1[:,ny//2:] = 1
    num_check = 4
    ch_x, ch_y = nx//num_check, ny//num_check
    mask2 = np.zeros((nx,ny))
    for x in range(nx):
      for y in range(ny):
        if( (x//ch_x + y//ch_y) %2 ==0):
          mask2[x,y] = 1
    mask3 = (mask1 + mask2)/2
    if num == 0:
      return mask1 
    elif num == 1:
      return mask2
    elif num == 2:
      return mask3

  def setTestcaseParameters(self, mask, regionSize, solverParamsFile ):
    # masked area is 80% si 20% TI,
    # unmasked area is 60% si 40% TI
    self.gds = GDSLoader(regionSize, solverParamsFile)
    k_si_mask = (0.8*mask + 0.6*(1-mask)) 
    k_sio2_mask =(0.2*mask + 0.4*(1-mask))
    self.gds.k_si = self.si_k*k_sio2_mask
    self.gds.k_sio2 =  self.sio2_k*k_sio2_mask
    self.gds.e = (1-self.r_sd)*k_si_mask + (1-self.r_ti)*k_sio2_mask
    self.grid.regionSize = regionSize*1e-6
    self.grid.gds = self.gds

