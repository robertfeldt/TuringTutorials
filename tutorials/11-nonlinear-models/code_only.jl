using Turing, Distributions
using DataFrames
using MCMCChains, Plots, StatsPlots
using Distances

using Random
Random.seed!(0)

data = DataFrame(R = Float64[0, 0.04, 0.07, 0.11, 0.165, 0.225, 0.27, 0.31, 0.315],
                 S = Float64[0,   30,   70,  140,   300,   600, 1200, 2600,  3800]);

michaelis_menten(b1, b2, s) = (b1 * s) / (b2 + s)

# Bayesian non-linear regression, Michaelis-Menten kinetics.
@model function nonlinear_regression_MM(R, S)
    # Set the priors on our coefficients.
    b1 ~ truncated(Gamma(1.5, 0.8),   0.001, Inf)
    #b2 ~ truncated(Gamma(1.5, 0.002), 0.001, Inf)
    b2 ~ Normal(500, 20)

    # Set variance prior. Default in brms is a Normal(0, 1)??
    σ ~ truncated(Normal(0, 1), 0, Inf)

    # Calculate all the mu terms.
    for i in eachindex(R)
        r = michaelis_menten(b1, b2, S[i]) # (b1 * S[i]) / (b2 + S[i]) # Michaelis-Menten kinetics model
        R[i] ~ Normal(r, sqrt(σ))
    end
end
                 
model = nonlinear_regression_MM(data.R, data.S)
# All of these fail:
chain = sample(model, NUTS(0.65), 4_000)
chain = sample(model, NUTS(20, 0.95), 4_000)
# Doesn't help:
#using DynamicHMC
#chain = sample(model, DynamicNUTS(), 4_000)
#chain = sample(model, IS(), 4_000)
chain = sample(model, HMCDA(200, 0.65, 0.3), 4_000)
# Doesn't work:
chain = sample(model, PG(), 4_000)


# Make predictions given the parameter values of the chain. Skip the initial half.
function prediction_mm(chain, S, numskip = div(length(chain), 2))
    b1vals = get(chain, :b1).b1[(numskip+1):end]
    b2vals = get(chain, :b2).b2[(numskip+1):end]
    medvals, qlows, qhighs = Float64[], Float64[], Float64[]
    for s in S
        preds = michaelis_menten.(b1vals, b2vals, s)
        qs = quantile(preds, [0.025, 0.50, 0.975])
        push!(qlows, qs[1])
        push!(medvals, qs[2])
        push!(qhighs, qs[3])
    end
    return medvals, qlows, qhighs
end

preds, qlows, qhighs = prediction_mm(chain, data.S)
msd(data.R, preds)

# Let's use BBO to find a good mode and then use that to set the prior on b2. We use OnlineStats
# to save the 5% quantile (since minimizing) and save all values found during optimization that
# are at least as good as it.
using BlackBoxOptim
const S = data.S
const R = data.R

# Minimize the mean squared distance, msd, while saving all param values that lead to top 5%
# fitness:
function fitness(x::Vector{Float64})
    preds = michaelis_menten.(x..., S)
    msd(preds, R)
end

# We assume b1 and b2 are positive and < 10k:
res = bboptimize(fitness; Method = :generating_set_search, SearchRange = (1e-10, 1000.0), NumDimensions = 2, MaxTime = 2.0);
modeb1, modeb2 = best_candidate(res) # 0.34, 298.3

# Now we can set a prior which is quite wide around the mode found:
@model function nonlinear_regression_MM2(R, S)
    # Set the priors on our coefficients.
    b1 ~ truncated(Normal(0.34, 0.20), 0.001, Inf)
    b2 ~ truncated(Normal(298.3, 50.0), 0.001, Inf)

    # Set variance prior. Default in brms is a Normal(0, 1)??
    σ ~ truncated(Normal(0, 1), 0, Inf)

    # Calculate all the mu terms.
    for i in eachindex(R)
        r = michaelis_menten(b1, b2, S[i]) # (b1 * S[i]) / (b2 + S[i]) # Michaelis-Menten kinetics model
        R[i] ~ Normal(r, sqrt(σ))
    end
end

model2 = nonlinear_regression_MM2(data.R, data.S)
chain = sample(model2, NUTS(0.65), 4_000)

# Another idea is to use BBO also to find the largest ranges of values that give distances
# within some fixed factor of the mode.
const BestFitness = msd(michaelis_menten.(modeb1, modeb2, S), R)
const FactorWorse = 10
function fitnessrange(x::Vector{Float64})
    b1a, b1b = x[1:2]
    b2a, b2b = x[3:4]
    predsa = michaelis_menten.(b1a, b2a, S)
    da = msd(predsa, R)
    predsb = michaelis_menten.(b1b, b2b, S)
    db = msd(predsb, R)

    # Penalty if msd distance is more than FactorWorse worse than best
    penaltya = (da > FactorWorse*BestFitness) ? 1000.0 + abs(da - FactorWorse*BestFitness) : 0.0
    penaltyb = (db > FactorWorse*BestFitness) ? 1000.0 + abs(db - FactorWorse*BestFitness) : 0.0

    # We want to maximize the size of the ranges so fitness is the negative of that (since minimizing overall)
    rangesize = abs(b1a - b1b) + abs(b2a - b2b)

    penaltya + penaltyb - 0.0001 * rangesize
end

res = bboptimize(fitnessrange; Method = :generating_set_search, SearchRange = (1e-10, 1000.0), NumDimensions = 4, MaxTime = 2.0);
# When I run this I get the best candidate something like: 0.370357, 0.314395, 440.648, 200.681
# So let's set Uniform(0.31, 0.38) and Uniform(200, 440) priors.

@model function nonlinear_regression_MM3(R, S)
    # Set the priors on our coefficients.
    #b1 ~ Uniform(0.31, 0.38)
    #b2 ~ Uniform(200.0, 440.0)
    b1 ~ Uniform(0.01, 1.0)
    b2 ~ Uniform(10.0, 500.0)

    # Set variance prior. Default in brms is a Normal(0, 1)??
    σ ~ truncated(Normal(0, 1), 0, Inf)

    # Calculate all the mu terms.
    for i in eachindex(R)
        r = michaelis_menten(b1, b2, S[i]) # (b1 * S[i]) / (b2 + S[i]) # Michaelis-Menten kinetics model
        R[i] ~ Normal(r, sqrt(σ))
    end
end

model3 = nonlinear_regression_MM3(data.R, data.S)
chain = sample(model3, NUTS(0.65), 4_000)
