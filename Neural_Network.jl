using CSV
using DataFrames, Random
using Flux, Statistics, ProgressMeter
using Plots
using Dates
include("Utils.jl")

type = "euler" # or "central_diff"
epochs = 20000
max_learning_rate = 0.01
ϵ=sqrt(eps(Float32))
l = 10.0 / 300.0
g = 10.0
c = 5.0
delta_t = 1.0e-3

df = DataFrame(CSV.File("output/$(type).csv", types=[Float32, Float32, Float32]))
# train has first 35% of the datapoints
train, test = splitdf(df, 0.35)

# select timesteps at an interval for complete dataset.
timesteps = select(df, ["timestep"])
interval = 50
physics_train_data = transpose(Matrix(timesteps[1:interval:size(timesteps)[1], :]))
a=physics_train_data.+ϵ
b=physics_train_data.-ϵ
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
prediction_per_epoch = []
@showprogress for epoch in 1:epochs
    pred_at_epoch = missing
    for (x, y) in loader
        loss, grad = Flux.withgradient(pars) do
            # Evaluate model and loss inside gradient context:
            y_hat = model(x)
            loss_mse = Flux.mse(y_hat, y)

            # compute the "physics loss"
            y_hat_physics = model(physics_train_data)
            phi_a = model(a)
            phi_b = model(b)
            pred_at_epoch=model(hcat(train_data,test_data))
            loss_residual = PhysicsLoss(phi_a,phi_b,y_hat_physics,ϵ, l, g,c)

            total_loss = loss_mse + loss_residual
            return total_loss
        end
        Flux.update!(opt, pars, grad)
        push!(losses, loss)  # logging, outside gradient context
    end
    push!(prediction_per_epoch, pred_at_epoch)
end



display(plot(losses; xaxis="iteration",
    yaxis="loss", label="per batch"))
savefig("output/nn_loss_$(now()).svg")

train_data_plot=collect(train_data[i] for i in 1:30:length(train_data))
train_target_plot=collect(train_target[i] for i in 1:30:length(train_target))
predictions = model(hcat(train_data,test_data))

plot(df.timestep, df.phi, xaxis="timestep(seconds)",
    yaxis="phi(radians)", label="solved with $(type) equation")
plot!(transpose(hcat(train_data,test_data)), transpose(predictions), label="PINNs prediction")
scatter!(train_data_plot,train_target_plot, label="training data")
# display(plot!(transpose(train_data), transpose(predictions[1:length(train_data)]), label="train predictions"))

savefig("output/nn_prediction_$(now()).svg")

prediction_animation = @animate for epoch ∈ 20:20:10000
    #plot(df.timestep, df.phi, label="solved with eq", title="Epoch: $(epoch)", ylim=(-0.4, 0.6))
    #plot!(transpose(hcat(train_data, test_data)), transpose(prediction_per_epoch[epoch]), label="predictions")
    plot(df.timestep, df.phi, xaxis="timestep(seconds)",
    yaxis="phi(radians)", label="solved with $(type) equation",title="Epoch: $(epoch)",ylim=(-0.4, 0.6))
    plot!(transpose(hcat(train_data,test_data)), transpose(prediction_per_epoch[epoch]), label="PINNs prediction")
    scatter!(train_data_plot,train_target_plot, label="training data")
    #frame(prediction_animation)
end
gif(prediction_animation, "output/nn_animation_60fps.gif", fps=10)
