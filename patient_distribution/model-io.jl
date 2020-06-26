using CSV, DataFrames, LinearAlgebra, SparseArrays

include("model.jl")

"""
This script reads data from `data.csv` and prints the solution to `solution.csv`.
"""
function model_io()
  df = CSV.read("data.csv")
  q⁰ = df.q0
  q = df[!,2:8]
  u = df.u
  P = [df.px df.py]
  n, nt = size(q)
  c = [norm(P[i,:] - P[j,:]) for i = 1:n, j = 1:n]

  x, y, z, obj, status = patient_distribution_model(q⁰, q, u, c)

  period      = Int[]
  origin      = Int[]
  destination = Int[]
  amount      = Int[]
  for t = 1:nt
    IJV = findnz(x[:,:,t])
    append!(origin, IJV[1])
    append!(destination, IJV[2])
    append!(amount, IJV[3])
    append!(period, fill(t, length(IJV[1])))
  end
  move_df = DataFrame(period=period, origin=origin, destination=destination, amount=amount)

  CSV.write("solution-move.csv", move_df)
  CSV.write("solution-occupation.csv", DataFrame(y.data, [Symbol("q$j") for j = 0:nt]))
  CSV.write("solution-new.csv", DataFrame(z, [Symbol("q$j") for j = 1:nt]))
end

model_io()