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
function run_step(int::Integrator, type::String, pendulum::Math_pendulum)
    if type == "euler"
        run_euler_step(int, pendulum)
    elseif type == "central_diff"
        run_central_diff_step(mp, pendulum)
    else
        println("... integration type not understood ...")
    end
end

## euler integration time step (homework)
function run_euler_step(int::Integrator, pendulum::Math_pendulum)
    # print(".")
    c = 5.0
    ###### (homework) ######
    omega_sq = pendulum.g / pendulum.l
    phi_dot_dot = -c * pendulum.phi_dot - omega_sq * pendulum.phi
    phi_i_plus_1 = pendulum.phi + int.delta_t * pendulum.phi_dot
    phi_dot_i_plus_1 = pendulum.phi_dot + int.delta_t * phi_dot_dot
    pendulum.phi = phi_i_plus_1
    pendulum.phi_dot = phi_dot_i_plus_1
end

## central difference time step (homework)
function run_central_diff_step(int::Integrator, pendulum::Math_pendulum)
    println("Running central difference step")
    ###### (homework) ######
end
