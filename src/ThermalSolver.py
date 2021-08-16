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

class ThermalSolver:
  def __init__(self):
    self.createLogger()
    self.ksiScaling = False

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

  def setParameters(self):
    nz = self.dz.shape[0]
    e = np.repeat(self.e[:,:,np.newaxis],nz,axis=2)
    e[:,:,1:] = 0
    k_si = np.repeat(self.k_si[:,:,np.newaxis],nz,axis=2)
    k_si[:,:,self.die_size:] = self.si_k
    k_sio2 = np.repeat(self.k_sio2[:,:,np.newaxis],nz,axis=2)
    k_sio2[:,:,self.die_size:] = 0
    self.k_si = k_si
    self.e = e
    self.k_sio2 = k_sio2

    self.width = self.e.shape[0]*self.region_size*1e-6
    self.length = self.e.shape[1]*self.region_size*1e-6
    self.dx = self.region_size*1e-6 # x dimension
    self.dy = self.region_size*1e-6 # y dimension

  def build(self, npzFile, regionSize, outDir):
    self.region_size = regionSize
    if outDir is not None:
      outDir.mkdir(parents=True, exist_ok=True)
    self.outDir=outDir
    st = time()
    self.gds = GDSLoader(regionSize, self.solverParamsFile)
    self.logger.log(25,"Creating emissivity map from GDS NPZ")
    self.gds.createEmmissivityMatrix(npzFile)
    self.e = self.gds.e
    self.k_si = self.gds.k_si
    self.k_sio2 = self.gds.k_sio2
    #self.defineParameters()
    self.setParameters()
    self.logger.log(25,"Created emissivity map in %5.2fs"%(time()-st))

  def buildTest(self, patternNum, regionSize, outDir):
    self.region_size = regionSize
    st = time()
    if outDir is not None:
      outDir.mkdir(parents=True, exist_ok=True)
    self.outDir=outDir
    self.logger.debug("Enabling pattern %d"%patternNum)
    patternMask = self.testcasePatterns(patternNum-1)
    #self.logger.debug(patternMask)
    #self.defineParameters()
    self.setTestcaseParameters(patternMask)
    self.setParameters()
      
    
  def runSolver(self, t_max, time_step, pw_lamp):
    self.t_lamp = pw_lamp
    self.t_step = time_step  # 0.1
    self.t_eval = np.arange(0,t_max,time_step)
    #self.logger.debug("t eval")
    #self.logger.debug(self.t_eval)
    nx,ny,nz = self.e.shape
    #gx, gy = np.mgrid[0:width:(nx*1j),0:length:(ny*1j)] #0.5 # emisivity
    T_init = self.T_bound*np.ones_like(self.e).reshape(-1)
    t1 = time()
    self.pbar = tqdm(total=t_max,position=0,leave=True,
      bar_format = "{desc}: {percentage:.1f}%|{bar}| {n:.1e}/{total:.1e} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    self.t_last = 0
    sol = solve_ivp(self.differential, 
                    [0, t_max], 
                    T_init, 
                    t_eval=self.t_eval, 
                    vectorized=True,
                    method = 'Radau') 
    temp = sol.y.T
    self.temp = temp.reshape(temp.shape[0], nx, ny, nz )

    self.logger.info("Completed solution in %5.2fs"%(time()-t1))
    self.pbar.close()
    self.saveData()
    #self.tempPlot()

  def differential(self,t,T):
    tdiff = t-self.t_last
    if tdiff>0:
      self.pbar.update(tdiff)
      self.t_last = t
    #self.logger.debug(T.shape)
    if len(T.shape) == 1:
      n_points = 0
      T = T.reshape(self.e.shape+(1,))
    else:
      n_points = T.shape[1]
      T = T.reshape(self.e.shape+(n_points,))

    area = self.dx * self.dy # 
    fxn = self.thermalConductivity(T)
    # scale fxn by 148 as k_si assumes the value to be 148
    k = self.k_sio2[...,np.newaxis] + (fxn/self.si_k) * self.k_si[...,np.newaxis] 
    # SiO2 1.4 W/m.K Si 148 W/m.K
    # i i+1  k1 T1
    k1 = np.zeros_like(k)
    T1 = np.zeros_like(T)
    k1[:-1,:,:,:] = k[1:,:,:,:]
    T1[:-1,:,:,:] = T[1:,:,:,:]
    k1[-1,:,:,:] = k[0,:,:,:]
    T1[-1,:,:,:] = T[0,:,:,:]
    k1 = 2/(1/k + 1/k1)

    # i i-1  k2 T2 
    k2 = np.zeros_like(k)
    T2 = np.zeros_like(T)
    k2[1:,:,:,:] = k[:-1,:,:,:]
    T2[1:,:,:,:] = T[:-1,:,:,:]
    k2[0,:,:,:] = k[-1,:,:,:]
    T2[0,:,:,:] = T[-1,:,:,:]
    k2 = 2/(1/k + 1/k2)

    # j j+1  k3 T3
    k3 = np.zeros_like(k)
    T3 = np.zeros_like(T)
    k3[:,:-1,:,:] = k[:,1:,:,:]
    T3[:,:-1,:,:] = T[:,1:,:,:]
    k3[:,-1,:,:] = k[:,0,:,:]
    T3[:,-1,:,:] = T[:,0,:,:]
    k3 = 2/(1/k + 1/k3)

    # j j-1  k4 T4 
    k4 = np.zeros_like(k)
    T4 = np.zeros_like(T)
    k4[:,1:,:,:] = k[:,:-1,:,:]
    T4[:,1:,:,:] = T[:,:-1,:,:]
    k4[:,0,:,:] = k[:,-1,:,:]
    T4[:,0,:,:] = T[:,-1,:,:]
    k4 = 2/(1/k + 1/k4)

    # add a section for k k-1
    # k5, k6, T5, T6
    # z z+1  k3 T3
    dz3d = self.dz[np.newaxis,np.newaxis,:,np.newaxis]
    k5 = np.zeros_like(k)
    T5 = np.zeros_like(T)
    dz5 = np.zeros_like(k)
    k5[:,:,:-1,:] = k[:,:,1:,:]
    T5[:,:,:-1,:] = T[:,:,1:,:]
    dz5[:,:,:-1,:]=dz3d[:,:,1:,:]
    k5[:,:,-1,:] = self.k_bound 
    T5[:,:,-1,:] = self.T_bound 
    dz5[:,:,-1,:] = self.dz_bound
    k5 =(dz3d+dz5)/(dz3d/k + dz5/k5)
    dz5 = (dz5+dz3d)/2

    # z z-1  k4 T4 
    k6 = np.zeros_like(k)
    T6 = np.zeros_like(T)
    dz6 = np.zeros_like(k)
    k6[:,:,1:,:] = k[:,:,:-1,:]
    T6[:,:,1:,:] = T[:,:,:-1,:]
    dz6[:,:,1:,:]=dz3d[:,:,:-1,:]
    k6[:,:,0,:] = self.k_bound
    T6[:,:,0,:] = self.T_bound
    dz6[:,:,0,:]= self.dz_bound
    k6 =(dz3d+dz6)/(dz3d/k + dz6/k6)
    dz6 = (dz6+dz3d)/2

    #self.logger.debug(" all shapes %s %s %s %s %s %s"%(T1.shape, T2.shape, T3.shape, T4.shape, T5.shape,
    #T6.shape))
    KT = ( (k1*(T1-T) + k2*(T2-T))/(self.dx*self.dx) 
         + (k3*(T3-T) + k4*(T4-T))/(self.dy*self.dy)
         + k5*(T5-T)/(dz5*dz5)
         + k6*(T6-T)/(dz6*dz6)
         )
    #self.logger.debug("KT shape %s"%(KT.shape,))

    q_t = self.sigma * self.e[...,np.newaxis] * area * (self.lampThermalProfile(t)**4 - T**4)
    #self.logger.debug("qt shape %s"%(q_t.shape,))
    dTdt_ = (1/(self.cp*self.density)) * ( KT  + q_t/(area*dz3d))
    #self.logger.debug("dtdt shape %s"%(dTdt_.shape,))
    if n_points==0:
      dTdt_ = dTdt_.reshape(-1)
    else:
      dTdt_ = dTdt_.reshape(-1,n_points)
    #self.logger.debug(dTdt_.shape)

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
    #fig.savefig('%s/T_Len_vs_Width.png'%(out_dir), bbox_inches='tight')
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

  def setTestcaseParameters(self, mask):
    # masked area is 80% si 20% TI,
    # unmasked area is 60% si 40% TI
    k_si_mask = (0.8*mask + 0.6*(1-mask)) 
    k_sio2_mask =(0.2*mask + 0.4*(1-mask))
    self.k_si = self.si_k*k_sio2_mask
    self.k_sio2 =  self.sio2_k*k_sio2_mask
    self.e = (1-self.r_sd)*k_si_mask + (1-self.r_ti)*k_sio2_mask


