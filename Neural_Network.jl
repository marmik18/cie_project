using CSV
using DataFrames, Random
using Flux, Statistics, ProgressMeter
using Plots
include("Utils.jl")

type = "euler" # or "central_diff"
epochs = 30000
max_learning_rate = 0.0001

l = 10.0 / 300.0
g = 10.0
omega_sq = g / l
c = 5.0

df = DataFrame(CSV.File("output/$(type).csv", types=[Float32, Float32, Float32]))
# train has first 35% of the datapoints
train, test = splitdf(df, 0.35)

# select timesteps at an interval for complete dataset.
timesteps = select(train, ["timestep"])
interval = 20
physics_train_data = transpose(Matrix(timesteps[1:interval:size(timesteps)[1], :]))

# select all fields for data and target
train_data = transpose(Matrix(select(train, ["timestep"])))
train_target = transpose(Matrix(select(train, ["phi"])))
# select all fields for data and target
test_data = transpose(Matrix(select(test, ["timestep"])))
test_target = transpose(Matrix(select(test, ["phi"])))

model = Chain(
    Dense(1 => 12, tanh),
    Dense(12 => 12, sigmoid),
    Dense(12 => 1, tanh),
)

out = model(train_data)
loader = Flux.DataLoader((train_data, train_target), batchsize=64, shuffle=true)
pars = Flux.params(model)
opt = Flux.AdaMax(max_learning_rate)

# println(pars)

# Training loop, using the whole data set 10 times:
losses = []
@showprogress for epoch in 1:epochs
    i=1
    for (x, y) in loader
        j=1
        loss, grad = Flux.withgradient(pars) do
            # Evaluate model and loss inside gradient context:
            y_hat = model(x)
            loss_mse = Flux.mse(y_hat, y)

            # compute the "physics loss"
            y_hat_physics = model(physics_train_data)

            # x(t) = sum(y_hat_physics ./ t)
            # dx(t) = gradient(x, t)[1]

            something = derivative(y_hat_physics, physics_train_data)
            # dxx = derivative(dx, physics_train_data)
            # print(dx)
            # dx(xt)

            # d2x(t) = gradient(dx, t)[1]
            # d2x(physics_train_data)
            # g = @diff sum(y_hat_physics ./ physics_train_data)
            # y_hat_physics_dot = grad(g, physics_train_data)
            
            # g_1 = @diff sum(y_hat_physics_dot ./ physics_train_data )
            # y_hat_physics_dot_dot = grad(g_1, physics_train_data)
            # y_hat_physics_dot = grad(y_hat_physics, physics_train_data)
            # y_hat_physics_dot_dot = grad(y_hat_physics, y_hat_physics_dot)
            # y_hat_physics_dot   = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0] # computes dy/dx
            # y_hat_physics_dot_dot = torch.autograd.grad(dx,  x_physics, torch.ones_like(dx),  create_graph=True)[0] # compu

            # physics = y_hat_physics_dot_dot + c * y_hat_physics_dot + omega_sq * y_hat_physics # computes the residual of the 1D harmonic oscillator differential equation
            # loss_physics = (1e-4) * mean(physics^2)

            # if epoch == 1 && i == 1  
            #     println(loss_physics)
            # end
            total_loss = loss_mse
            return total_loss
        end
        j+=1
        Flux.update!(opt, pars, grad)
        push!(losses, loss)  # logging, outside gradient context
    end
    i+=1
end

plot(losses; xaxis=(:log10, "iteration"),
    yaxis="loss", label="per batch")
n = length(loader)
plot!(n:n:length(losses), mean.(Iterators.partition(losses, n)),
    label="epoch mean", dpi=200)

# savefig("output/$(type)_loss.png")

predictions_train_target = model(train_data)
predictions = model(test_data)

plot(df.timestep, df.phi, xaxis="timestep",
    yaxis="phi", label="solved with $(type) equation")
plot!(transpose(test_data), transpose(predictions), label="predictions")
plot!(transpose(train_data), transpose(train_target), label="train")
plot!(transpose(train_data), transpose(predictions_train_target), label="train predictions")

# savefig("output/$(type)_prediction.png")
