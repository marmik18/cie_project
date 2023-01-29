using Flux: Zygote

# split dataframe by percent and return train and test dataframe
function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    # shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end

function PhysicsLoss(phi_a, phi_b, phi, ϵ, l, g, c)
    # Calculate physics loss
    loss = []
    for i in 1:(length(phi))
        phi_dot = (phi_a[i] - phi_b[i]) / (2 * ϵ)
        phi_dotdot = (phi_a[i] - 2 * phi[i] + phi_b[i]) / (ϵ)^2
        r = phi_dotdot + c * phi_dot + g / l * phi[i]
        loss = vcat(loss, [r])
    end
    return (1e-4) * mean(loss .^ 2)
end