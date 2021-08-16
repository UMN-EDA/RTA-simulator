import numpy as np
from tqdm import tqdm, trange
from pathlib import Path
import logging
from time import time

class GDSLoader():
  def __init__(self, region_size):
    self.createLogger()
    self.region_size_um = region_size
    self.k_si = None
    self.k_siO2 = None
    self.e = None
    self.k = None
    
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
    self.logger.debug(emm_full.shape)
    emm_full[RTA_array==0] = 0.80
    emm_full[RTA_array==1] = 0.43
    emm_full[RTA_array==2] = 0.55
    
    k_sio2_full[RTA_array==0] = 1.4
    k_si_full[RTA_array==1] = 148
    k_si_full[RTA_array==2] = 148

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
