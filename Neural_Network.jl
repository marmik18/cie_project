using CSV
using DataFrames, Random
using Flux, Statistics, ProgressMeter
using Plots
using Dates
include("Utils.jl")

type = "euler" # or "central_diff"
epochs = 15000
max_learning_rate = 0.01

l = 10.0 / 300.0
g = 10.0
c = 5.0
delta_t = 1.0e-3

df = DataFrame(CSV.File("output/$(type).csv", types=[Float32, Float32, Float32]))
# train has first 35% of the datapoints
train, test = splitdf(df, 0.35)

# select timesteps at an interval for complete dataset.
timesteps = select(train, ["timestep"])
interval = 50
physics_train_data = transpose(Matrix(timesteps[1:interval:size(timesteps)[1], :]))

# select all fields for data and target
train_data = transpose(Matrix(select(train, ["timestep"])))
train_target = transpose(Matrix(select(train, ["phi"])))
# select all fields for data and target
test_data = transpose(Matrix(select(test, ["timestep"])))
test_target = transpose(Matrix(select(test, ["phi"])))

model = Chain(
    Dense(1 => 32, tanh),
    Dense(32 => 32, sigmoid),
    Dense(32 => 1),
)

out = model(train_data)
loader = Flux.DataLoader((train_data, train_target), batchsize=64, shuffle=true)
pars = Flux.params(model)
opt = Flux.Adam(max_learning_rate)

# Training loop, using the whole data set 10 times:
losses = []
prediction_animation = Animation()
@showprogress for epoch in 1:epochs
    for (x, y) in loader
        loss, grad = Flux.withgradient(pars) do
            # Evaluate model and loss inside gradient context:
            y_hat = model(x)
            loss_mse = Flux.mse(y_hat, y)

            # compute the "physics loss"
            y_hat_physics = model(hcat(train_data, test_data))

            loss_residual = PhysicsLoss(y_hat_physics, l, g, c, delta_t)

            total_loss = loss_mse + loss_residual
            return total_loss
        end
        Flux.update!(opt, pars, grad)
        push!(losses, loss)  # logging, outside gradient context
    end
    y_hat_physics = model(hcat(train_data, test_data))
    plot(df.timestep, df.phi, label="solved with eq", title="Epoch: $(epoch)")
    plot!(transpose(hcat(train_data, test_data)), transpose(y_hat_physics), label="predictions")
    frame(prediction_animation)
end



plot(losses; xaxis="iteration",
    yaxis="loss", label="per batch")
savefig("output/nn_loss_$(now()).png")

predictions_train_target = model(train_data)
predictions = model(test_data)

plot(df.timestep, df.phi, xaxis="timestep",
    yaxis="phi", label="solved with $(type) equation")
plot!(transpose(test_data), transpose(predictions), label="predictions")
plot!(transpose(train_data), transpose(train_target), label="train")
plot!(transpose(train_data), transpose(predictions_train_target), label="train predictions")

savefig("output/nn_prediction_$(now()).png")
gif(prediction_animation, "output/nn_animation_60fps.gif", fps=60)
