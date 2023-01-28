using Flux: Zygote

# split dataframe by percent and return train and test dataframe
function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    # shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end

function PhysicsLoss(phi, l, g, c, delta_t)

    # Calculate loss as the mean squared error (MSE) between the predicted and target values
    loss = []
    stepsize = 1
    for i in 1+stepsize:50:length(phi)-stepsize
        phi_dot = (phi[i+stepsize] - phi[i-stepsize]) / (2 * delta_t * stepsize)
        phi_dotdot = (phi[i+stepsize] - 2 * phi[i] + phi[i-stepsize]) / (delta_t * stepsize)^2
        r = phi_dotdot + c * phi_dot + g / l * phi[i]
        loss = vcat(loss, [r])
    end
    # for i in 1+stepsize:length(phi)-stepsize
    #     phi_dot = (phi[i+stepsize] - phi[i-stepsize]) / (2 * delta_t * stepsize)
    #     phi_dotdot = (phi[i+stepsize] - 2 * phi[i] + phi[i-stepsize]) / (delta_t * stepsize)^2
    #     r = phi_dotdot + c * phi_dot + g / l * phi[i]
    #     loss = vcat(loss, [r])
    # end

    return (5e-4) * mean(loss .^ 2)
end

