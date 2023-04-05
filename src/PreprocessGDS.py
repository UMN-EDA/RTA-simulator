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
import math
import numpy as np
from gdsii import types
from gdsii.record import Record
import json
from tqdm import tqdm
from pathlib import Path
import logging
from matplotlib import pyplot as plt

class PreprocessGDS():
  def __init__(self):
    self.createLogger()
  def createLogger(self):
    self.logger = logging.getLogger('TAZ.PRE')

  def generateJson(self, GdsFile):
    self.logger.log(25,"Generating JSON file from GDS file %s"%GdsFile)
    self.GdsFile = GdsFile
    self.GdsJson = self.convert_GDS_GDSjson()

  def saveGdsJson(self, outFile):
    if self.GdsJson == None:
      self.logger.error("GDS json object does not exist, saving failed.")
      return
    outpath = Path(outFile)
    if outpath.parent.is_dir():
      self.logger.log(25,"Saving JSON file to %s."%outpath.absolute())
      with outpath.open('w') as f:
        json.dump (self.GdsJson, f, indent=2)
    else:
      self.logger.error("GDS json outfile parent folder does not exist, saving failed.")
    return
  
  def loadGdsJson(self, JsonFile):
    with JsonFile.open('r') as f:
      self.GdsJson = json.load(f)

  def defineParameters(self, solverParamsFile):
    self.solverParamsFile = solverParamsFile
    with solverParamsFile.open('r') as f:
      solver_params = json.load(f)
    self.layer_types = solver_params['layer_types']
    self.layer_mapping = solver_params['layer_mapping']


  def createNpzFromGDS(self, GdsFile, outDir, highRes, averageFile):
    self.generateJson(GdsFile)
    self.generateNpz(outDir,highRes,averageFile)
  
  def createNpzFromJson(self, JsonFile, outDir):
    self.loadGdsJson(JsonFile)
    self.generateNpz(outDir)
  
  def calcAvgEmmissivity(self, design_map, averageFile, design_name):
    if averageFile is not None:
      avg_emm = 0
      for l_type, layer in self.layer_types.items():
        if l_type == "default":
          continue
        l = int(l_type) 
        r = layer['r'] 
        avg_emm += (1-r) * np.sum(design_map==l)/design_map.size
      with averageFile.open("a") as f:
        f.write("%s, %5.4f\n"%(design_name, avg_emm))

  def transform(self, inst_map_in, strans, angle, row, col, dx, dy, limits, name):
    if(col>1):
      if(dx!=0):
        inst_map_row = np.tile(inst_map_in[0:dx,:],(col-1,1))
      else:
        inst_map_row = np.tile(inst_map_in        ,(col-1,1))
      inst_map_row = np.concatenate((inst_map_row, inst_map_in),axis=0)
    else:
      inst_map_row = inst_map_in
    if(row>1):
      if(dy!=0):
        inst_map = np.tile(inst_map_row[:,0:dy],(1,row-1))
      else:
        inst_map = np.tile(inst_map_row        ,(1,row-1))
      inst_map = np.concatenate((inst_map, inst_map_row),axis=1)
    else:
      inst_map = inst_map_row

    w, h = inst_map.shape
    (minx,maxx,miny,maxy) = limits 
    maxx = minx+w
    maxy = miny+h
    if strans == 32768: #reflection about x axis
      inst_map = np.flip(inst_map, 1) # flip along y axis
      miny, maxy= -maxy, -miny
    elif strans == 0: # do nothing
      pass
    else:
      self.logger.warning("Unknown Strans condition for %s: %x"%(name, strans))
    if int(angle) == 0:
      pass
    elif int(angle) == 90:
      minx, maxx, miny, maxy = -maxy, -miny, minx, maxx 
      inst_map = np.rot90(inst_map,-1) # as we are working with x,y
    elif int(angle) == 180:
      minx, maxx, miny, maxy = -maxx, -minx, -maxy, -miny 
      inst_map = np.rot90(inst_map,2)
    elif int(angle) == 270:
      minx, maxx, miny, maxy = miny, maxy, -maxx, -minx 
      inst_map = np.rot90(inst_map,1) # as we are working with x,y
    else:  
      self.logger.warning("Unknown angle condition for %s: %x"%(name, angle))

    return inst_map, (minx, maxx, miny, maxy)

  def generateNpz(self, outDir, highRes, averageFile):
    self.logger.log(25,"Generating GDS NPZ.")
    plot = False
    gds = self.GdsJson
    outDir.mkdir(parents=True, exist_ok=True)
    
    if highRes:
      res = 0.001 # 0.1um
    else:
      res = 0.1
    layout = gds['bgnlib'][0]
    unit = layout['units'][0]/ res
    
    designs = {}
    #TODO: sub designs must be defined before they are used. 
    for design in tqdm(layout['bgnstr']):
      design_name = design['strname']
      if( ('elements' not in design) or
          len(design['elements']) <=0):
        self.logger.warning("Skipping design %s as it does not contain any elements."%design_name)
        continue
      designs[design_name] = {}
      designs[design_name]['layers'] = {}
      designs[design_name]['instances']= []
      for element in  design['elements']:
        if element['type'].lower() == 'boundary'.lower():
          layer = element['layer']
          if layer not in designs[design_name]['layers']:
            designs[design_name]['layers'][layer] = []
          xy = np.array(element['xy']).reshape((-1,2))
          min_x, max_x = np.min(xy[:,0]),np.max(xy[:,0])
          min_y, max_y = np.min(xy[:,1]),np.max(xy[:,1])
          designs[design_name]['layers'][layer].append(
              (min_x, max_x, min_y, max_y))
        elif (element['type'].lower() == 'aref'.lower() or
          element['type'].lower() == 'sref'.lower()): 
          instance = {}
          inst_name = element['sname']
          instance['name'] = inst_name
          instance['strans'] = element.get('strans', 0)
          instance['angle'] = element.get('angle', 0)
            
          if element['type'].lower() == 'aref'.lower():
            instance['col'] = element['colrow'][0]
            instance['row'] = element['colrow'][1]
          else:
            instance['col'] = 1
            instance['row'] = 1
          if instance['name'] not in designs:
            self.logger.warning("%s not processed before %s. This may lead to map size issues"%(instance['name'],design_name))

          xy = np.array(element['xy']).reshape((-1,2))
          x, y = int(xy[0,0]*unit),int(xy[0,1]*unit)
          #assuming no stadard 90 rotation or mirroring
          dx  = math.ceil((max(xy[:,0]) - min(xy[:,0]))*unit/instance['col'])
          dy  = math.ceil((max(xy[:,1]) - min(xy[:,1]))*unit/instance['row'])
          instance['dx'], instance['dy']  = dx, dy

          inst_map = designs[inst_name]['layer_map']
          inst_map, limits = self.transform(inst_map,
                                    instance['strans'],
                                    instance['angle'],
                                    instance['row'],
                                    instance['col'],
                                    instance['dx'],
                                    instance['dy'],
                                    designs[inst_name]['limits'],
                                    inst_name)
          (minx,maxx,miny,maxy) = limits
          instance['min_x'] = x+limits[0]
          instance['max_x'] = x+limits[1]
          instance['min_y'] = y+limits[2]
          instance['max_y'] = y+limits[3]
          designs[design_name]['instances'].append(instance)      
    
      layer_map= {}    
      maxx, maxy =-float('inf'), -float('inf')
      minx, miny =float('inf'), float('inf')
      for  layer in designs[design_name]['layers'].values():
        boundaries = np.array(layer)
        min_x = int(np.min(boundaries[:,0])*unit) 
        max_x = int(np.max(boundaries[:,1])*unit)
        min_y = int(np.min(boundaries[:,2])*unit)
        max_y = int(np.max(boundaries[:,3])*unit)
        maxx,maxy = max(maxx,max_x), max(maxy,max_y)
        minx,miny = min(minx,min_x), min(miny,min_y)
      for inst in designs[design_name]['instances']:
        maxx = max(maxx,inst['max_x'])
        maxy = max(maxy,inst['max_y'])
        minx = min(minx,inst['min_x'])
        miny = min(miny,inst['min_y'])
        
      for l_name, layer in tqdm(designs[design_name]['layers'].items(),leave=False):
        boundaries = np.array(layer)
        layer_map[l_name] = np.zeros((maxx-minx,maxy-miny),dtype='byte')
        for element in boundaries:
          x1, x2, y1, y2 =  ((element*unit).astype('int')
                             - np.array([minx,minx,miny,miny]))
          layer_map[l_name][x1:x2+1,y1:y2+1] = 1
    
      full_map = (np.ones((maxx-minx,maxy-miny),dtype='byte')
                 )* self.layer_types['default']
      for l in self.layer_mapping:
        if int(l) in layer_map and self.layer_mapping[l] != 0:
          full_map[layer_map[int(l)] == 1] = self.layer_mapping[l]

      designs[design_name]['layer_map'] = full_map
      designs[design_name]['limits'] = (minx,maxx,miny,maxy)
    
    self.logger.log(25,"Saving GDS NPZ files to %s."%outDir)
    for design_name,design in tqdm(designs.items(), position=0, leave=True):
      if len(design['instances'])>0:
        layer_map = design['layer_map']
        integrate_map = np.zeros_like(layer_map)
        for instance in tqdm(design['instances'], position=1, leave=True):
          inst_name = instance['name']
          if (inst_name not in designs):
            self.logger.warning("Skipping instance %s as it was not preprocessed."%inst_name)
            continue
          if 'integrated' not in designs[inst_name]:
            self.logger.error("Block %s was not integrated before it was needed."%inst_name)
            pass
          else:
            pass
            
          inst_map = designs[inst_name]['layer_map']
          inst_map, limits = self.transform(inst_map,
                                    instance['strans'],
                                    instance['angle'],
                                    instance['row'],
                                    instance['col'],
                                    instance['dx'],
                                    instance['dy'],
                                    designs[inst_name]['limits'],
                                    inst_name)
          (minx,maxx,miny,maxy) = designs[design_name]['limits'] 
          llx = instance['min_x'] - minx
          urx = instance['max_x'] - minx
          lly = instance['min_y'] - miny 
          ury = instance['max_y'] - miny
          integrate_area = integrate_map[llx:urx,lly:ury]
          integrate_area[inst_map>0] = inst_map[inst_map>0]
          integrate_map[llx:urx,lly:ury] = integrate_area
        integrate_map[layer_map>0] = layer_map[layer_map>0]
        design_map = integrate_map
        design['layer_map'] = integrate_map
        design['integrated'] = True
      elif len(design['instances']) == 0:
        design_map = design['layer_map']
        design['integrated'] = True
      self.calcAvgEmmissivity(design_map, averageFile, design_name)
      np.savez_compressed('%s/%s.npz'%(outDir,design_name),design=design_map,resolution=np.array([res])) 
      if plot:
        plt.figure()
        plt.imshow(design_map.T)
        plt.title('Layout map %s'%design_name)
        plt.colorbar()
    if plot:
      plt.show()
####### This is derived from an open source code for gds to json conversion
#  This helps understand:  http://www.buchanan1.net/stream_description.html
#  element:  boundary | path | sref | aref | text | node | box
  def show_data(self, rec):
    """Shows data in a human-readable format."""
    data = rec.data
    tag_type = rec.tag_type
    if tag_type == types.ASCII:        return "%s" % data.decode()
    elif tag_type == types.BITARRAY:   return int(data)
    elif tag_type == types.REAL8:
        if len(data) > 1:              return [float(s) for s in data]
        else:                          return data[0]
    elif tag_type == types.INT2 or tag_type == types.INT4:
        if len(data) > 1:              return [int(s) for s in data]
        else:                          return data[0]
    return '\"' + ', '.join('{0}'.format(i) for i in data) + '\"'

  def isElement (self,e):
    return e == "BOUNDARY" or e == "PATH" or e == "SREF" or \
           e == "AREF" or e == "TEXT" or e == "NODE" or e == "BOX"

  def convert_GDS_GDSjson (self):
    level = 0


    top = {}
    cursors = [top, {}, {}, {}, {}, {}, {}]

    with self.GdsFile.open('rb') as a_file:
      for rec in tqdm(Record.iterate(a_file)):
        tag_name = rec.tag_name
        tag_type = rec.tag_type

        jsonName = ""
        if self.isElement (tag_name):          jsonName = "elements"
        else:                             jsonName = tag_name.lower()

        if ((tag_type != types.NODATA and tag_name[0:3] == "BGN") or
              self.isElement(tag_name)):
          if isinstance(cursors[level], dict):
            level = level + 1
            cursors[level] = []
            cursors[level - 1][jsonName] = cursors[level]
          level = level + 1
          cursors[level] = {}
          cursors[level - 1].append (cursors[level])

          if self.isElement(tag_name): cursors[level]["type"] = tag_name.lower()
          if tag_name[0:3] == "BGN": cursors[level]["time"] = self.show_data (rec)

        if tag_type != types.NODATA and tag_name[0:3] != "BGN":
          cursors[level][jsonName] = self.show_data (rec)
        elif tag_name[0:3] == "END":
          if isinstance(cursors[level - 1], dict): level = level - 1
          level = level - 1

    return top
