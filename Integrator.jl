####################################
# Explicit Euler
#
# numeric integration file for the
# mathematical pendulum
#
# - explicit euler
# -
####################################


mutable struct Integrator
    delta_t::Float64
    timesteps::Int64
    Integrator(delta_t, timesteps) = new(delta_t, timesteps)
    res_phi::Vector
    res_phi_dot::Vector
end

## run one integration time step
function run_step(int::Integrator, type, pendulum)
    if type == "euler"
        run_euler_step(int, pendulum)
    elseif type == "central_diff"
        run_central_diff_step(mp, pendulum)
    else
        println("... integration type not understood ...")
    end
end

## euler integration time step (homework)
function run_euler_step(int::Integrator, pendulum)
    println("Running euler step")
    ###### (homework) ######
end

## central difference time step (homework)
function run_central_diff_step(int::Integrator, pendulum)
    println("Running central difference step")
    ###### (homework) ######
end
