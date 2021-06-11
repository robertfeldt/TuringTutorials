# Load Turing.
using Turing
# Load other dependencies
using Distributions, LinearAlgebra
using VegaLite, DataFrames

# Example data set - generate synthetic gene expression data
n_cells = 50
n_genes = 18
mu_1 = 10. * ones(n_genes÷3)
mu_0 = zeros(n_genes÷3)
S = I(n_genes÷3)
mvn_0 = MvNormal(mu_0, S)
mvn_1 = MvNormal(mu_1, S)

# create a diagonal block like expression matrix, with some non-informative cells
expression_matrix = transpose(vcat(hcat(rand(mvn_1, n_cells÷2), rand(mvn_0, n_cells÷2)),
                                   hcat(rand(mvn_0, n_cells÷2), rand(mvn_0, n_cells÷2)),
                                   hcat(rand(mvn_0, n_cells÷2), rand(mvn_1, n_cells÷2))))


df_exp = convert(DataFrame, expression_matrix)
df_exp[!,:cell] = 1:n_cells
#  DataFrames.stack(df_exp, 1:n_genes) |> @vlplot(:rect, "cell:o", "variable:o", color=:value) |> save("expression_matrix.pdf")
DataFrames.stack(df_exp, 1:n_genes) |> @vlplot(:rect, "cell:o", "variable:o", color=:value)

@model pPCA(x, ::Type{T} = Float64) where {T} = begin

  # Dimensionality of the problem.
  N, D = size(x)

  # latent variable z
  z = Matrix{T}(undef, D, N)
  for n in 1:N
    z[:, n] ~ MvNormal(D, 1.)
  end

  # weights/loadings w
  w = Matrix{T}(undef, D, D)
  for d in 1:D
    w[d, :] ~ MvNormal(D, 1.)
  end

  # mean offset
  mean = Vector{T}(undef, D)
  mean ~ MvNormal(D, 1.0)
  mu = w * z .+ mean

  for d in 1:D
    x[:,d] ~ MvNormal(mu[d,:], 1.)
  end

end

ppca = pPCA(expression_matrix)

# Hamiltonian Monte Carlo (HMC) sampler parameters
n_iterations = 500
ϵ = 0.05
τ = 10

#  It is important to note that although the maximum likelihood estimates of W,\mu in the pPCA model correspond to the PCA subspace, only posterior distributions can be obtained for the latent data (points on the subspace). Neither the mode nor the mean of those distributions corresponds to the PCA points (orthogonal projections of the observations onto the subspace). However what is true, is that the posterior distributions converge to the PCA points as \sigma^2 \rightarrow 0. In other words, the relationship between pPCA and PCA is a bit more subtle than that between least squares and regression.
chain = sample(ppca, HMC(ϵ, τ), n_iterations)

describe(chain)[1]

# Extract paramter estimates for plotting - mean of posterior
w = permutedims(reshape(mean(group(chain, :w))[:,2], (n_genes,n_genes)))
z = permutedims(reshape(mean(group(chain, :z))[:,2], (n_genes, n_cells)))'
mu = mean(group(chain, :mean))[:,2]

X = w * z .+ mu

df_rec = convert(DataFrame, X')
df_rec[!,:cell] = 1:n_cells

#  #  DataFrames.stack(df_rec, 1:n_genes) |> @vlplot(:rect, "cell:o", "variable:o", color=:value) |> save("reconstruction.pdf")
DataFrames.stack(df_rec, 1:n_genes) |> @vlplot(:rect, "cell:o", "variable:o", color=:value)

df_pro = DataFrame(z')
rename!(df_pro, Symbol.( ["z"*string(i) for i in collect(1:n_genes)]))
df_pro[!,:cell] = 1:n_cells

DataFrames.stack(df_pro, 1:n_genes) |> @vlplot(:rect, "cell:o", "variable:o", color=:value)

df_pro[!,:type] = repeat([1, 2], inner = n_cells÷2)
df_pro |>  @vlplot(:point, x=:z1, y=:z2, color="type:n")

