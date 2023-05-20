# Orifice plate simulation
- Since everything is a repeating pattern, I can abuse that to my heart's content.
- We will simply simulate one orifice unit in many different configurations.

## Orifice configs
- Flow velocities
  - Keep mdot_cold = 0.7
  - Keep mdot_hot = 0.5
  - Vary number of tubes (per pass): 4,6,8,10,15,20
  - Divide mdot accordingly
- ~~Tube Pitch?~~ Eh probably not.
  - 14-20mm
- Baffle Pitch
  - 10, 20, 30, 40mm
- Orifice size
  - 1, 1.5, 2 mm
- Fixed length of 250mm
- 12 different geometries. Use a script to automate the mdot variation.


## Scrappy notes
- `triSurf().writeFms("C:/Users/Nik/Documents/OpenFOAM-sims/GA3/out.fms")`
