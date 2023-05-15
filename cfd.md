# Motivation
- A lot of the correlations used for the software are rather dodgy.
- Most obviously, the shell-side pressure drop correlations are very poor.
  - The other correlations are a little more believable.
- It might be worth running a batch of CFD simulations to obtain the mass flow rate for different shell configurations.
  - I.e. (2,3,4 baffles) x (10-20 tubes) = 33 configurations
  - Note that the number of tube passes has no effect on the shell hydraulics

# Automation
- I am ok with manually creating 33 `.stl` files.
- I am not ok with manually creating 33 meshes.
- I am not ok with manually post-processing 33 results.
- Therefore need to automate those processes.

## `.stl` creation
- Ensure the `.stl` is only of the shell-side geometry, and is appropriately labelled
- Option 1: SALOME
  - At 5 mins per `.stl`, this will take me 2 hrs 45 mins.
  - Not fun.
- Option 2: Blender
  - Could probably do 2 mins per `.stl`, meaning I'll be done in just over 1 hr.
  - Fine.

## Meshing
- `cfMesh` should be able to handle this with no troubles.
- Add a bunch of inflation layers to ensure good functioning of k-omega model.

## Simulation
- Use RANS k-omega turbulence model
- Use `simpleFoam`
- Solvers:
  - `p`: `GAMG`, `GaussSeidel` smoother
  - `u`: `smoothSolver`, `GaussSeidel`
- Solve until mass flow rate is converged

## Post Processing
- Simply output mass flow rate to console.

# ~~Heat transfer~~
## Summary
- I do not have the time nor inclination to set up a simulation workflow for this (much less, set it up for OpenFOAM)
- I trust the heat transfer coefficient correlations
- Hence it is not worth doing heat transfer CFD.
- Below is a discussion of the specifics that would / would not need to be considered

## Solid conduction
- We can neglect the thermal resistance of copper.
  - From example, resistance of copper is 1/447252
  - Resistances of convection are 1/6787 and 1/7933
  - Overall heat transfer coeff 
    - 3628 (including copper resistance)
    - 3657 (excluding copper resistance)
    - difference is less than 1%.

## Turbulence model
- Need turbulent thermal diffusivity model for water
- k-omega is well validated for such internal flows. Could also use k-omega SST.
  - Would use a laminar heat transfer coefficient at the wall (since we're resolving down to the viscous sublayer)
- LES is completely unnecessary in this case.

## Meshing
- Ideally, would model the baffles as either infinitely thin walls, or fairly thick walls.
- Need some way of connecting the inner and outer surfaces of the tubes with a boundary condition, and simulate both surfaces at once.
- Could do it with baffles.

## Decoupling
- Perhaps it is best to decouple the hot and cold sides? And simply approximate the boundary condition with a constant ambient temperature.
- In that case, it is pointless to simulate the tube side, since the correlations there are well supported by experiments.
- Therefore, only simulate the shell side, and see what heat transfer coefficient can be obtained.
- Then, it is a very simple simulation to run, although the results are a bit dodgy.

## No decoupling
- It would be nice to have a fully coupled simulation.
- Can use AMI to achieve this
  - https://www.cfd-online.com/Forums/openfoam-solving/152729-coupled-patches-heat-transfer.html
  - https://www.cfd-online.com/Forums/openfoam/107363-chtmultiregionfoam-different-mesh-2-sides-coupled-boundary.html
  - https://www.cfd-online.com/Forums/openfoam-solving/172293-coupling-patches-chtmultiregionsimplefoam.html
  - Not quite sure how to write the BC, even if we use `mappedWall`. 
  - Possibly someone else has run into this issue?
- Can use `nearestPatchFace`
- Now can mesh things a lot easier.

## Automation / iteration
- Manually create `.STL`s for the trial geometries.
- Create a script to do a full simulation given an `.STL`, and output the heat transfer rate.
- Iterate over pipe count, baffle count.
