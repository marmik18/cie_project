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
function run_step(int::Integrator, type::String, pendulum::Math_pendulum, i::Int)
    if type == "euler"
        run_euler_step(int, pendulum)
    elseif type == "central_diff"
        run_central_diff_step(int, pendulum, i)
    else
        println("... integration type not understood ...")
    end
end

## euler integration time step (homework)
function run_euler_step(int::Integrator, pendulum::Math_pendulum)
    # print(".")
    ###### (homework) ######
    omega_sq = pendulum.g / pendulum.l
    phi_dot_dot = -(pendulum.c / pendulum.m) * pendulum.phi_dot - omega_sq * pendulum.phi
    phi_i_plus_1 = pendulum.phi + int.delta_t * pendulum.phi_dot
    phi_dot_i_plus_1 = pendulum.phi_dot + int.delta_t * phi_dot_dot
    pendulum.phi = phi_i_plus_1
    pendulum.phi_dot = phi_dot_i_plus_1
end

## central difference time step (homework)
function run_central_diff_step(int::Integrator, pendulum::Math_pendulum, i::Int)
    dt = int.delta_t
    length = pendulum.l
    mass = pendulum.m
    gravitation = pendulum.g
    friction = pendulum.c

    # Setting u0 / u0_dot  
    u0 = pendulum.phi
    u0_dot = pendulum.phi_dot
    M_inverse = 1 / (mass*length^2)
    u0_dot_dot = M_inverse * (0 - friction*pendulum.phi_dot - mass*gravitation*length*sin(pendulum.phi))

    

    ## Calculation phi
    u0 - dt * u0_dot + dt ^ 2 * u0_dot_dot


    ## Calculation phi_dot_dot
    

end
