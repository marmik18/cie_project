using Flux: Zygote

# split dataframe by percent and return train and test dataframe
function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    # shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end

function derivative(x, t)
    input = deepcopy(x)
    time = deepcopy(t)
    
    limit = size(input)[2]

    res = zeros(size(input))

    for i in 1:limit
        if i == limit
            res[i] = input[i] / time[i] - time[i-1]
        else
            res[i] = (input[i+1] - input[i]) / time[i+1] - time[i]
        end
    end

    return copy(res)
end

