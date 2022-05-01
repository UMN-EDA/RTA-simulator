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

class Visualizer():
  def __init__(self):
    self.createLogger()

  def createLogger(self):
    self.logger = logging.getLogger('TAZ.VIS')

  def visualizeEmmissivity(self,npzFile,regionSize, solverParamsFile):
    gds = GDSLoader(regionSize, solverParamsFile)
    st = time()
    self.logger.log(25,"Creating emissivity map from GDS NPZ")
    gds.createEmmissivityMatrix(npzFile)
    plt.figure()
    plt.imshow(gds.e.T, cmap='jet')
    plt.title("Emissivity map at %d um resolution"%regionSize)
    plt.xlabel('Width (um)')
    plt.ylabel('Length (um)')
    self.logger.info("Average emissivity: %5.3f"% np.mean(gds.e))
    plt.colorbar()
    self.logger.log(25,"Created emissivity map in %5.2fs"%(time()-st))

  def loadSolutionFile(self,solFile, outDir):
    solution = np.load(solFile)
    self.temp = solution['temp']
    self.t_lamp = solution['t_lamp']
    self.width = solution['width']
    self.length = solution['length']
    self.dz = solution['dz']
    self.t_eval = solution['t_eval']
    if outDir is not None:
      outDir.mkdir(parents=True, exist_ok=True)
    self.outDir=outDir

  def visualizeLvW(self,t_point=None):
    if t_point is None:
      t_point = self.t_lamp
    nt,nx,ny,nz = self.temp.shape
    t_step = self.t_eval[1] - self.t_eval[0]
    t_plot_point = min(int(t_point/t_step),nt-1)
    T_lw_plot = self.temp[t_plot_point,:,:,0].squeeze()
    len_plot = int(100* self.length/self.width)
    y_ticks = (
                [y for y in np.arange(0,len_plot+1,0.2*len_plot)],
                [int(self.length*y*1e6/100) for y in np.arange(0,101,20)],
              )
    x_ticks = (
                [x for x in np.arange(0,101,20)],
                [int(self.width*x*1e6/100) for x in np.arange(0,101,20)],
              )
    title="Length vs Width plot at %5.2es"%t_point
    self.plot_im(plot_data=T_lw_plot.T,
                 x_axis= "Width (um)",
                 y_axis= "Length (um)",
                 x_ticks=x_ticks,
                 y_ticks=y_ticks,
                 len_plot=len_plot,
                 title=title,
                 save_name='LengthVsWidth.png'
                 )
 
  def visualizeLvH(self, t_point):
    if t_point is None:
      t_point = self.t_lamp
    nt,nx,ny,nz = self.temp.shape
    t_step = self.t_eval[1] - self.t_eval[0]
    t_plot_point = min(int(t_point/t_step),nt-1)
    T_height_plot =self.temp[t_plot_point,int(nx//2),:,:].squeeze()

    len_plot = int(200)
    y_ticks = (
                [y for y in np.arange(0,len_plot+1,0.2*len_plot)],
                [int(self.length*y*1e6/100) for y in np.arange(0,101,20)],
              )
    x_ticks = (
                [x for x in np.arange(0,101,100/len(self.dz))],
                ["%2.1f"%(np.sum(self.dz[:z])*1e6) for z in np.arange(len(self.dz)+1)],
              )
    title="Length vs Height plot at %5.2es"%t_point
    self.plot_im(plot_data=T_height_plot,
                 x_axis= "Height (um)",
                 y_axis= "Length (um)",
                 x_ticks=x_ticks,
                 y_ticks=y_ticks,
                 len_plot=len_plot,
                 title=title,
                 save_name='LengthVsHeight.png'
                 )

  def visualizeTvt(self):
    t_step = self.t_eval[1] - self.t_eval[0]
    t_max = self.t_eval[-1]
    nt,nx,ny,nz = self.temp.shape
    temp = self.temp[:,int(nx//2),int(ny//2),0].squeeze()
    temp_time = np.stack((self.t_eval, temp))
    max_temp = temp_time.max()

    len_plot = int(100)
    x_ticks = (
                [        (t_max*x/100)     for x in np.arange(0,101,20)],
                ["%3.1f"%(t_max*x*1e3/100) for x in np.arange(0,101,20)]
              )
    y_ticks = (
                [        (max_temp*y/100) for y in np.arange(0,101,20)],
                ["%3.1f"%(max_temp*y/100) for y in np.arange(0,101,20)]
              )
    title="Temperature vs Time plot"
    self.plot_line(plot_data=temp_time,
                 x_axis= "Time (ms)",
                 y_axis= "Temperature (K)",
                 x_ticks=x_ticks,
                 y_ticks=y_ticks,
                 len_plot=len_plot,
                 title=title,
                 save_name='TemperatureVsTime.png'
                 )

  def visualizeLvT(self):
    t_step = self.t_eval[1] - self.t_eval[0]
    t_max = self.t_eval[-1]
    nt,nx,ny,nz = self.temp.shape
    temp_length = self.temp[:,int(nx//2),:,0].squeeze()

    len_plot = 50
    y_ticks = (
                [y for y in np.arange(0,len_plot+1,0.2*len_plot)],
                [int(self.length*y*1e6/100) for y in np.arange(0,101,20)],
              )
    x_ticks = (
                [x for x in np.arange(0,101,20)],
                ["%3.1f"%(t_max*x*1e3/100) for x in np.arange(0,101,20)],
              )
    title="Length vs Time plot"
    self.plot_im(plot_data=temp_length.T,
                 x_axis= "Time (ms)",
                 y_axis= "Length (um)",
                 x_ticks=x_ticks,
                 y_ticks=y_ticks,
                 len_plot=len_plot,
                 title=title,
                 save_name='LengthVsTime.png'
                 )

  def plot_line(self, plot_data, x_axis, y_axis, x_ticks, y_ticks,
              title, len_plot,save_name):
    fig,ax = plt.subplots()
    width_plot = 100
    im = ax.plot(plot_data[0], plot_data[1])
    ax.set_yticks(y_ticks[0])
    ax.set_yticklabels(y_ticks[1])
    ax.set_xticks(x_ticks[0])
    ax.set_xticklabels(x_ticks[1])
    ax.set_ylabel(y_axis)
    ax.set_xlabel(x_axis)
    ax.set_title(title, pad=20)
    fig.set_size_inches(len_plot*0.09,width_plot*0.09) # convert to inches, 100->4 inches
    if self.outDir is not None:
      fig.savefig(self.outDir / save_name, bbox_inches='tight',dpi=100)

  def plot_im(self, plot_data, x_axis, y_axis, x_ticks, y_ticks,
              title, len_plot,save_name):
    fig,ax = plt.subplots()
    width_plot = 100
    im = ax.imshow(plot_data , cmap = 'jet', extent= [0,width_plot, len_plot, 0])
    ax.set_yticks(y_ticks[0])
    ax.set_yticklabels(y_ticks[1])
    ax.set_xticks(x_ticks[0])
    ax.set_xticklabels(x_ticks[1])
    ax.set_ylabel(y_axis)
    ax.set_xlabel(x_axis)
    ax.set_title(title, pad=20)
    fig.colorbar(im)
    fig.set_size_inches(len_plot*0.09,width_plot*0.09) # convert to inches, 100->4 inches
    if self.outDir is not None:
      fig.savefig(self.outDir / save_name, bbox_inches='tight',dpi=100)
