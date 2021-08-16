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
from tqdm import tqdm, trange
from pathlib import Path
import logging
from time import time
import json

#Class for reading GDS to convert into the NPZ files
class GDSLoader():
  def __init__(self, region_size, solverParamsFile):
    self.createLogger()
    self.region_size_um = region_size
    self.k_si = None
    self.k_siO2 = None
    self.e = None
    self.k = None
    self.defineParameters(solverParamsFile)

  def defineParameters(self, solverParamsFile):
    with solverParamsFile.open('r') as f:
      solver_params = json.load(f)
    self.si_k = solver_params['si_k'] 
    self.sio2_k = solver_params['sio2_k'] 
    self.r_sd = solver_params['r_sd'] 
    self.r_ti = solver_params['r_ti'] 
    self.r_gate = solver_params['r_gate'] 

  def createLogger(self):
    self.logger = logging.getLogger('TAZ.GDS')

  def createEmmissivityMatrix(self, PreprocessedGdsFile):
    gdsPath = Path(PreprocessedGdsFile)
    if not gdsPath.is_file():
      self.logger.error("File %s, does not exists"%PreprocessedGdsFile)
      return
    self.logger.info("Loading file %s"%PreprocessedGdsFile)
    GDS_npz = np.load(PreprocessedGdsFile)
    RTA_array = GDS_npz['design']
    resolution = GDS_npz['resolution']
    self.logger.info("Loading file %s complete"%PreprocessedGdsFile)
    region_size  =  int(float(self.region_size_um)/resolution)
    if region_size> RTA_array.shape[0] or region_size > RTA_array.shape[1] :
      self.logger.error("Region_size is larger that the chip")
    if region_size < resolution :
      self.logger.error("Region_size is lesser that resolution")
    nx, ny = [int(x/region_size) for x in RTA_array.shape]
    emissivity_array = np.zeros((nx,ny))
    conductivity_array_sio2 = np.zeros((nx,ny))
    conductivity_array_si = np.zeros((nx,ny))
    num_el = (RTA_array.shape[0]*RTA_array.shape[1])/(region_size*region_size)
    ##
    self.logger.debug("Creating emissivity and conductivity maps")
    emm_full = np.zeros_like(RTA_array,dtype=float)
    k_si_full = np.zeros_like(RTA_array,dtype=float)
    k_sio2_full = np.zeros_like(RTA_array,dtype=float)
    self.logger.debug("Emissivity map shape %s"%(emm_full.shape,))
    emm_full[RTA_array==0] = (1-self.r_ti)
    emm_full[RTA_array==1] = (1-self.r_sd)
    emm_full[RTA_array==2] = (1-self.r_gate)
    
    k_sio2_full[RTA_array==0] = self.sio2_k
    k_si_full[RTA_array==1] = self.si_k
    k_si_full[RTA_array==2] = self.si_k

    self.logger.debug("Aggregating info")
    emissivity_array = np.sum(emm_full.reshape(nx, region_size, 
                                               ny, region_size),
                              axis=(1,3)
                             )/(region_size**2)
    conductivity_array_si = np.sum(k_si_full.reshape(nx, 
                              region_size, ny, region_size),
                              axis=(1,3)
                             )/(region_size**2)
    conductivity_array_si = np.sum(k_sio2_full.reshape(nx, 
                              region_size, ny, region_size),
                              axis=(1,3)
                             )/(region_size**2)

    self.logger.debug("Aggregating info complete")
    self.e = emissivity_array
    self.k_si = conductivity_array_si
    self.k_sio2 = conductivity_array_sio2
    self.k = conductivity_array_si + conductivity_array_sio2
