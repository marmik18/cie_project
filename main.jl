## running Euler integrator for the mathematical pendulum
######################################
######################################
######################################

# Juno.clearconsole()

using PrettyTables
using Plots, Printf
using DelimitedFiles

## booleans
show_video = false

## starting the file
println("-- pendulum euler --")

## load pendulum
include("Dynsys.jl")
pendulum = Dynsys.Math_pendulum(1.0, 10.0, 1.0, 0.5, 0.0)

## load integrator and memory for the results
Integ = Dynsys.Integrator(1.0e-3,5)
Integ.res_phi = zeros(Integ.timesteps)
Integ.res_phi_dot = zeros(Integ.timesteps)

## run time integration
# initial setting
fig = Dynsys.create_fig(pendulum)
Dynsys.plot_state(pendulum)
display(fig)
# running over the time step
for i in 1:Integ.timesteps
    # integration step
    # (homework)
    # plot the state
    fig = Dynsys.create_fig(pendulum)
    Dynsys.run_step(Integ, "euler", pendulum)
    Dynsys.plot_state(pendulum)
    display(fig)
    # save the step
    Integ.res_phi[i] = pendulum.phi
    Integ.res_phi_dot[i] = pendulum.phi_dot
end

######## Homework
# implement the euler integration step
# implement the central difference integration step
# plot coordinate phi and the time derivative phi_dot
