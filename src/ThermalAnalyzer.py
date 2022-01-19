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
import sys
import argparse
import logging
from PreprocessGDS import PreprocessGDS
from Visualizer import Visualizer
from ThermalSolver import ThermalSolver 
from pathlib import Path
from matplotlib import pyplot as plt
from time import time

class ThermalAnalyzer:
  def __init__(self):
    self.preprocessGds = PreprocessGDS()
    self.visualizer = Visualizer()
    self.simulator = ThermalSolver()

  def run(self,):
    self.parseInputArguments()
    if self.args.AnalysisMode== 'preprocessGDS':
      self.PreprocessGdsOptions()
        
    elif self.args.AnalysisMode== 'simulate':
      self.SimulateOptions()
    elif self.args.AnalysisMode== 'visualize':
      self.VisualizeOptions()

  
  def PreprocessGdsOptions(self):
    
    if self.args.jsonFile is not None:
      self.modeJsonPreprocess()
    elif self.args.gdsFile is not None:
      self.modeGdsPreprocess()
    else:
      self.logger.critical("Undefined state for Prepocessign GDS.")
  
  def VisualizeOptions(self):
    if self.args.npzFile is not None:
      if self.args.R is None:
        self.parserVisualize.error("--resolution is a required argument with --emissivity")
        return 
      if not self.args.solverParams.is_file():
        self.logger.error('''Solver Params file %s does not exist, Please define
        it for creating emissivity maps'''%self.args.solverParams)
        return
      self.logger.info("Reading definition file %s"%self.args.solverParams)
      self.visualizer.visualizeEmmissivity(self.args.npzFile, self.args.R, self.args.solverParams)
    if (self.args.lvt is not False or 
        self.args.lvh is not False or
        self.args.lvw is not False ):
      if self.args.solFile is None:
        self.parserVisualize.error('''--solution is a required argument with -lvt,
                                      -lvh, -lvw''')
        return
      if not self.args.solFile.is_file():
       self.logger.error('''Solution file does not exist, please provide a valid
                           file to --solution''')
       return
      self.visualizer.loadSolutionFile(self.args.solFile, self.args.outDir)
      if self.args.lvw:
        self.visualizer.visualizeLvW(self.args.t_point)
      if self.args.lvh:
        self.visualizer.visualizeLvH(self.args.t_point)
      if self.args.lvt:
        self.visualizer.visualizeLvT()
    plt.show()


  def SimulateOptions(self):
    self.logger.debug("t_max %e"%self.args.t_max)
    self.logger.debug("t_step %e"%self.args.t_step)
    self.logger.debug("pw_lamp %e"%self.args.pw_lamp)
    self.logger.debug("tr_lamp %e"%self.args.tr_lamp)
    self.logger.debug("tf_lamp %e"%self.args.tf_lamp)
    if self.args.solverParams.is_file():
      self.logger.info("Reading definition file %s"%self.args.solverParams)
      self.simulator.defineParameters(self.args.solverParams)
    else:
      self.logger.error('''Solver Params file %s does not exist. Please define
      it to run the simulator'''%self.args.solverParams)
      return
    if self.args.tr_lamp > self.args.pw_lamp:
      self.logger.error('''Lamp rise time cannot be larger than the pulse width.
      The pulse width time includes the lamp rise time.''')
      return
    if self.args.tr_lamp < 0 or self.args.tf_lamp <0:
      self.logger.error('''Lamp rise time and fall time must be positive 
      numbers.''')
      return

    if self.args.npzFile is not None:
      self.simulator.build(self.args.npzFile, self.args.R, self.args.outDir, 
                           self.args.halfMode, self.args.discretization)  
    elif self.args.test is not None:
      self.simulator.buildTest(self.args.test, self.args.R, self.args.outDir,
                               self.args.discretization)  
    self.simulator.enableKsiScaling(self.args.ksiScaling)
    self.simulator.runSolver(self.args.t_max, self.args.t_step, self.args.pw_lamp,
                             self.args.tr_lamp, self.args.tf_lamp)
      
  def modeJsonPreprocess(self):
    self.logger.critical("Reading from JSON is not yet implmented.")
    #outDir = self.args.outDir
    #jsonFile = self.args.jsonFile
    #self.preprocessGds.createNpzFromJson( jsonFile, outDir)


  def modeGdsPreprocess(self):
    outDir = self.args.outDir
    gdsFile = self.args.gdsFile
    self.preprocessGds.createNpzFromGDS( gdsFile, outDir)

    
    
  def create_logger(self, log_file=None,severity=None):
    # Create a custom logger
    logging.addLevelName(25,'STATUS')
    logger = logging.getLogger('TAZ')
    # Create handlers
    c_handler = logging.StreamHandler()
    if severity is None:
      logger.setLevel('STATUS')
      c_handler.setLevel('STATUS')
    else:
      logger.setLevel(severity)
      c_handler.setLevel(severity)
    # Create formatters and add it to handlers
    c_format = logging.Formatter('[%(name)s][%(levelname)s] %(message)s')
    c_handler.setFormatter(c_format)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    # handle log file
    if log_file is not None:
      f_handler = logging.FileHandler(str(log_file.absolute()))
      if severity is None:
        f_handler.setLevel('INFO')
      else:
        f_handler.setLevel(severity)
      f_format = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(message)s]')
      f_handler.setFormatter(f_format)
      logger.addHandler(f_handler)
    return logger

  def parseInputArguments(self):
    parser = argparse.ArgumentParser(prog = "ThermalAnalyzer",
              description = '''Thermal simulation and analysis for rapid 
                               thermal anealing (RTA).''')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true",
              help = "Enables verbose mode with detailed information.")
    group.add_argument("-q", "--quiet", action="store_true",
              help = '''Supresses all informational messages and only
                        displays warnings and errors.''')
    group.add_argument("-d", "--debug", action="store_true",
              help = '''Displays additional debug messages for 
                        software debug. Warning: This can lead to 
                        significant increase in messages.''')
    parser.add_argument("-l", "--log_file", type=Path,
              help = "Log file for run.",dest='log')
    parser.set_defaults(func=lambda : parser.print_help())                        
    source_dir = Path(__file__).resolve().parent.parent
    config_file = source_dir / 'config/SolverParams.json'
    parser.add_argument("-p",'--solver_params', type=Path,
                              dest='solverParams', default=config_file,
                              help = ''' JSON file containing necessary
                              definition to run the tool. By default loads
                              %s'''%config_file)
    subparsers = parser.add_subparsers(
              title = "ThermalAnalyzer subcommands and analysis modes.",
              description="List of available modes for %(prog)s", 
              help=''' 
              For additional information of each subcommand run 
              %(prog)s <subcommand> --help''',
              dest='AnalysisMode')
    subparsers.required = True
    parserSim = subparsers.add_parser("simulate", 
              description="Thermal simualation mode",
              help=''' Setup the tool in thermal simulation mode to run analysis
              on the provided GDS.
              ''')
    group_sim = parserSim.add_mutually_exclusive_group(required=True)
    group_sim.add_argument("-g","--preprocessedGDS", type=Path, dest='npzFile',
                           action = 'append', help = '''Path to the preprocessed
                              NPZ file for simulation. Input each file with a
                              separate -g option''')

    group_sim.add_argument("-t","--testcase",type=int,dest='test',
                            choices=[1,2,3], help =''' Run predefined
                            testcases''')
    parserSim.add_argument("-r","--resolution", type=int, dest='R', required=True,
                              help = ''' Lateral resolution in um (int) for simulation''')
    parserSim.add_argument("-tm","--time_max", type=float, dest='t_max', required=True,
                              help = ''' Maximum time for the simulation''')
    parserSim.add_argument("-ts","--time_step", type=float, dest='t_step', required=True,
                              help = ''' Time step resolution for the simulation''')
    parserSim.add_argument("-tp","--time_pulse", type=float, dest='pw_lamp', required=True,
                              help = ''' Time duration of the lamp pulse for the simulation''')
    parserSim.add_argument("-trl","--time_rise_lamp", type=float, dest='tr_lamp', default=0.0,
                              help = ''' Time for the lamp to switch on and rise to the
                              maximum temperature''')
    parserSim.add_argument("-tfl","--time_fall_lamp", type=float, dest='tf_lamp', default=0.0,
                              help = ''' Time for the lamp to switch off and
                              fall to the ambient maximum temperature''')
    parserSim.add_argument("-o",'--outDir', type=Path, dest='outDir',
                              help = ''' Destination directory for output
                              solution files. The command will create the directory
                              if it does not exists''')
    parserSim.add_argument("-k",'--enable_t_k_scaling', action='store_true',
                              dest='ksiScaling', default=False,
                              help = ''' Enable temperature based scaling of
                              thermal conductivity of silicon. By default, this is
                              disabled''')
    parserSim.add_argument("-c",'--enable_coarsening', action='store_true',
                              dest='discretization', default=False,
                              help = ''' Enable coarsening of emissivity map to
                              speed up thermal simmulation. By default, this is
                              disabled''')
    parserSim.add_argument("-hm","--half_mode", default=False,
                            action="store_true", dest='halfMode',                          
                            help = ''' Process GDS in half mode where only the
                            first half of the GDS (width wise) is used ''')
    #### Visualize options
    parserVisualize = subparsers.add_parser("visualize", 
              description="GDS visualization",
              help=''' Set of suboptions for visualizing the inputs/outputs ''')
    group_emmis = parserVisualize.add_argument_group('Emmissivty',
                          'Necessary arguments to generate emissivity plots')
    group_emmis.add_argument("-e","--emissivity", type=Path, dest='npzFile',
                              help = '''Path to the preprocessed NPZ file for
                                        emissivty plot''')
    group_emmis.add_argument("-r","--resolution", type=int, dest='R',
                              help = '''Resolution in um (int) for emissivity plot''')
    parserVisualize.add_argument('-o','--outDir', type=Path, dest='outDir',
              help = '''Destination directory to store the figure. The command
              will create the directory if it does not exists''')
    group_temp = parserVisualize.add_argument_group('Temperature plots',
                          'Necessary arguments to generate temperature plots')
    group_temp.add_argument("-t","--time_point", type=float, dest='t_point',
                              help = '''Time point at which to plot the result''')
    group_temp.add_argument("-lvw","--lenVwidth", action="store_true",
                              dest='lvw', help = '''Plot length vs width for the 
                              provided time point''')
    group_temp.add_argument("-lvt","--lenVtime", action="store_true",
                              dest='lvt', help = '''Plot length vs time for the
                              along the center of the design''')
    group_temp.add_argument("-lvh","--lenVheight", action="store_true",
                              dest='lvh', help = '''Plot length vs height for the 
                              provided time point along the center of the design''')
    group_temp.add_argument("-s","--solution", type=Path, dest='solFile',
                              help = '''Path to the generated solution file from
                              simulate''')


    #### preprocess GDS options
    parserGdsPreprocess = subparsers.add_parser("preprocessGDS", 
              description="GDS pre-processing ",
              help=''' Preprocess the GDSII/JSON file to a
              preprocessed NPZ file that can be used by the other subcommands ''')
    group_preproc = parserGdsPreprocess.add_mutually_exclusive_group(required=True)
    group_preproc.add_argument("-j", "--json", type=Path, dest='jsonFile',
              help = "Load GDS information from a JSON file.")
    group_preproc.add_argument("-g", "--gds",  type=Path, dest='gdsFile',
              help = 'Load GDS information from a GDSII file.')
    parserGdsPreprocess.add_argument('-o','--outDir', type=Path,
              dest='outDir', required=True,
              help = '''Destination directory for output NPZ files. The command
              will create the directory if it does not exists''')
    self.parser = parser
    self.parserVisualize = parserVisualize
    self.parserGdsPreprocess = parserGdsPreprocess
    args = parser.parse_args()
    if args.verbose:
      severity = 'INFO'
    elif args.quiet:
      severity = 'WARNING'
    elif args.debug:
      severity = 'DEBUG'

    else:
      severity = None
    self.logger = self.create_logger( log_file = args.log,
                                      severity = severity)
    self.args = args

if __name__ == "__main__":
  st = time()
  therm_anlyzr = ThermalAnalyzer()
  therm_anlyzr.run()
  et =  time()
  t = et-st
  h = t//3600
  m = (t%3600)//60
  s = t%60
  print("TOTAL RUN TIME: %02d:%02d:%05.3f"%(h,m,s))
