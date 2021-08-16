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
from gdsii import types
from gdsii.record import Record
import json
from tqdm import tqdm
from pathlib import Path
import logging

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

  def createNpzFromGDS(self, GdsFile, outDir):
    self.generateJson(GdsFile)
    self.generateNpz(outDir)
  
  def createNpzFromJson(self, JsonFile, outDir):
    self.loadGdsJson(JsonFile)
    self.generateNpz(outDir)

  def generateNpz(self, outDir):
    self.logger.log(25,"Generating GDS NPZ.")
    plot = False
    gds = self.GdsJson
    outDir.mkdir(parents=True, exist_ok=True)

    res = 0.1 # 0.1um
    layout = gds['bgnlib'][0]
    unit = layout['units'][0]/ res
    
    designs = {}
    for design in tqdm(layout['bgnstr']):
      design_name = design['strname']
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
        elif element['type'].lower() == 'aref'.lower(): 
          instance = {}
          xy = np.array(element['xy']).reshape((-1,2))
          min_x, max_x = int(np.min(xy[:,0])*unit),int(np.max(xy[:,0])*unit) 
          min_y, max_y = int(np.min(xy[:,1])*unit),int(np.max(xy[:,1])*unit)
          instance['min_x'] = min_x
          instance['max_x'] = max_x
          instance['min_y'] = min_y
          instance['max_y'] = max_y
          instance['col'] = element['colrow'][0]
          instance['row'] = element['colrow'][1]
          instance['name'] = element['sname']
    
          designs[design_name]['instances'].append(instance)      
    
      layer_map= {}    
      maxx, maxy =0,0
      minx, miny =float('inf'), float('inf')
      for l_name, layer in designs[design_name]['layers'].items():
        boundaries = np.array(layer)
        min_x = int(np.min(boundaries[:,0])*unit) 
        max_x = int(np.max(boundaries[:,1])*unit)
        min_y = int(np.min(boundaries[:,2])*unit)
        max_y = int(np.max(boundaries[:,3])*unit)
        maxx,maxy = max(maxx,max_x), max(maxy,max_y)
        minx,miny = min(minx,min_x), min(miny,min_y)
        
      for l_name, layer in tqdm(designs[design_name]['layers'].items(),leave=False):
        boundaries = np.array(layer)
        layer_map[l_name] = np.zeros((maxx-minx,maxy-miny),dtype='byte')
        for element in boundaries:
          x1, x2, y1, y2 =  ((element*unit).astype('int')
                             - np.array([minx,minx,miny,miny]))
          layer_map[l_name][x1:x2+1,y1:y2+1] = 1
    
      full_map = np.zeros((maxx-minx,maxy-miny),dtype='byte')
      for l in [(2011,1),(2012,1), (2310,2), (2344,2)]:
        if l[0] in layer_map:
          full_map[layer_map[l[0]] == 1] = l[1]
      designs[design_name]['layer_map'] = full_map
      designs[design_name]['limits'] = (minx,maxx,miny,maxy)
    
    self.logger.log(25,"Saving GDS NPZ files to %s."%outDir)
    for design_name,design in tqdm(designs.items(), position=0, leave=True):
      if len(design['instances'])>0:
        layer_map = design['layer_map']
        integrate_map = np.zeros_like(layer_map)
        for instance in tqdm(design['instances'], position=1, leave=True):
          inst_name = instance['name']
          inst_map = designs[inst_name]['layer_map']
          inst_map = np.tile(inst_map,(instance['col'],instance['row']))
          (minx,maxx,miny,maxy) = designs[design_name]['limits'] 
          w,h = inst_map.shape  
          llx, lly = instance['min_x'] - minx , instance['min_y'] - miny
          urx, ury = llx + w, lly +h
          integrate_map[llx:urx,lly:ury] = inst_map
        integrate_map[layer_map>0] = layer_map[layer_map>0]
    
        if plot:
          plt.figure()
          plt.imshow(integrate_map.T)
          plt.title('integrated layout map %s'%design_name)
        np.savez_compressed('%s/%s.npz'%(outDir,design_name),design=integrate_map,resolution=np.array([res])) 
      elif len(design['instances']) == 0:
        if plot:
          plt.figure()
          plt.imshow(design['layer_map'].T)
          plt.title('layout map %s'%design_name)
        np.savez_compressed('%s/%s.npz'%(outDir,design_name),design=design['layer_map'],resolution=np.array([res])) 
    
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

    #ofile = open (oname, 'wt')

    top = {}
    cursors = [top, {}, {}, {}, {}, {}, {}]

    #with open(self.GdsFile, 'rb') as a_file:
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

    #json.dump (top, ofile, indent=4)
    return top
