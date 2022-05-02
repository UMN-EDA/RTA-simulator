# Readme - SolverParams.json

SolverParams is the main configuration files that defines all technology specific parameters. 
For a specific technology this only needs to be set once at the begining. 

## Parameters 

### dz
Description: dz describes the discretization in the z dimension into the substrate. 
Usage: Defines the height of each element in the solver.

Example: In the code snippet below the dimension of the z direction is 8 and the height of each element is defined by the value in the dz array. The total height of the substrate is the sum of all the dz elements (500e-6 in the example.)
```python
"dz": [100e-9,100e-9,100e-9,100e-9,600e-9,99e-6,100e-6,300e-6]
```

### die_depth
Description: die_depth indicates how many elements of dz belong to the die 
Usage: Defines the number of elements of dz that correspond to the die. The die will be populated with the conductivity of the GDS with the remaining being mapped to an Si substrate

Example: The first 4 elements of dz correspond to the die.
```python
"die_depth": 4
```

### Emmissivity discretization threshold
Description: Emmissivity threshold below which nodes can be merged during coarsening with -c.
Usage: The coarsening option is used to merge adjecent nodes in the solver to speed up the simulation. As emmisivity can have small variation within a local area, this threshold defines the maximum difference of emissivity between adjacent nodes that can safely be ignored, or within tolerable limits.

Example: Emmissivity threshold set as 5e-3.
```python
"emm_discretization_thresh": 5e-3,
```

### Lamp Properties
| Parameter     | Description                                 | Example|
| ------------- | ------------------------------------------- | -----: |
| temp_lamp_on  | temperature (in K) of the RTA lamp when on  |   3000 |
| temp_lamp_off | temperature (in K) of the RTA lamp when off |    298 |

### Material Properties
| Parameter     | Description                                 | Example|
| ------------- | ------------------------------------------- | -----: |
| density       |  Si density in kg/m^3                       |   2330 |
| cp            | Specific heat of Si J/(kg K)                |    700 |
| si_k          | Thermal conductivity of Si (W/mK) at room temp     | 148    |
| sio2_k        | Thermal conductivity of SiO2 (W/mK) at room temp   | 1.4    |
| sigma         | Stefan Boltzman constant                    | 5.67e-8 |

### Boundary conditions towards ambient
| Parameter     | Description                                 | Example|
| ------------- | ------------------------------------------- | -----: |
| T_bound       | Temperautre (K) of the regions surrounding the die in z direction  |   298 |
| k_bound       | Effective thermal conductivity (W/mK ) from the boundary of the die in z direction to heat sink |    0.1|
| dz_bound      | Effective distance (m) to the heat sink of the surroundings in z direction  | 0.5e-3|

Note: dz is multipled by k_bound to get the effective thermal conductance to the heat sink. and T_bound defines the temperature of the surrounding regions.

### Layer types in GDS
This section of the SolverParams.json sets up the GDS parser. This section consists of two parts. A layer type and a layer mapping.  

#### Layer Types
Defining the layers for the RTA-simulator. The layers must be defined to an integer mapping as shown below. The r values represents the emmisivity or reflectivity of the layer being defined. The m values defines the type of material (Si or Si02) being used.
Note: The tool will directly map layer with m=0 to thermal conductivity of silicon (Si) and m=1 to thermal conductivity of silicon dioxide (SiO2). The tool only supports conducitivies of Si and Sio2 as materials due to the tight integration of temperature dependent conductivity. Accounting for other materials would need a code rewrite. So currently other m values are not supported. 

```
 "layer_types": { 
    "0": {
      "Desription": "Trench isolation layer, reflection coefficient and material",
      "r": 0.2,
      "m": 0
    },
    "1": {
      "Desription": "Source Drain layer, reflection coefficient and material",
      "r": 0.57,
      "m": 1
    },
    "2": {
      "Desription": "Gate layer, reflection coefficient and material",
      "r": 0.45,
      "m": 1
    },
    "default": 0
  }, 
```

#### Layer mapping

 This section maps the layers in the GDS (2011, 2012, 2310, 2344) to the layers defined in the above section. 
 
 ``` 
 "layer_mapping": { 
    "2011": 1,
    "2012": 1,
    "2310": 2,
    "2344": 2
  },
 ```



