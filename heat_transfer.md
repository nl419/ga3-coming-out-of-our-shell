- Add up individual resistances
  - Convection coefficient on tube side
  - Convection coefficient on shell side
  - Conduction coefficient through tube

# Convection (tube side)
- Nu_i = 0.023 Re^0.8 Pr^0.3
- h_i = Nu_i * k_water / D_inner

# Convection (shell side)
- Nu_o = c Re^0.6 Pr^0.3
- c = 0.2 (triangular pitch)

# CFD?
## Solid conduction
- Copper is quite conductive.
- From example, resistance of copper is 1/447252
- Resistances of convection are 1/6787 and 1/7933
- Overall heat transfer coeff 
  - 3628 (including copper resistance)
  - 3657 (excluding copper resistance)
  - difference is less than 1%.
- So if we were to do CFD, we could quite comfortably neglect solid conduction, which would simplify things.

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
