## running Euler integrator for the mathematical pendulum
######################################
######################################
######################################

# Juno.clearconsole()

using PrettyTables
using Plots, Printf
using DelimitedFiles

## booleans
show_video = true
pendulum_animation = Animation()

## starting the file
println("-- pendulum euler --")

## load pendulum
include("Dynsys.jl")
pendulum = Dynsys.Math_pendulum(10.0 / 300.0, 10.0, 1.0, 0.5, 0.0)

## load integrator and memory for the results
delta_t = 1.0e-3
timesteps = 2000
type = "euler" # or "central_diff"

Integ = Dynsys.Integrator(delta_t, timesteps)
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
    Dynsys.run_step(Integ, type, pendulum)
    Dynsys.plot_state(pendulum)
    display(fig)
    frame(pendulum_animation)
    # save the step
    Integ.res_phi[i] = pendulum.phi
    Integ.res_phi_dot[i] = pendulum.phi_dot
end

x = LinRange(0, timesteps * delta_t, timesteps)
display(plot(x, Integ.res_phi, xlabel="time", ylabel="position"))
savefig("output/$(type)_position.png")

display(plot(x, Integ.res_phi_dot, xlabel="time", ylabel="velocity"))
savefig("output/$(type)_velocity.png")

if show_video
    gif(pendulum_animation, "output/$(type)_pendulum.gif")
end
######## Homework
# Done: implement the euler integration step
# implement the central difference integration step
# plot coordinate phi and the time derivative phi_dot
