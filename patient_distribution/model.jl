using Cbc, JuMP

"""
    x, y, z, obj, status = patient_distribution_model(q⁰, q, u, c)

Optimizes the patient allocation given the initial state of hospitals
`q⁰`, the predicted demand `q` and the hospital limits `u`.

Sets
- i, j ∈ H = {1,…,n}: hospitals.
- t ∈ T = {1,…,nt}: time frame.

Parameters
-  q⁰[j]: Starting allocation, i.e., how many people at Hospital j ∈ H.
- q[j,t]: Predicted demand at time t ∈ T at Hospital j ∈ H.
-   u[j]: Maximum capacity of Hospital j ∈ H .
- c[i,j]: Cost of moving one person from Hospital i to j, where i, j ∈ H.

Variables
- x[i,j,t]: Patients that need to be sent from Hospital i to j at instant t, where i, j ∈ H, t ∈ T.
-   y[j,t]: Total amount of patients admitted at Hospital j at instant t, where j ∈ H, t ∈ T.
    y[j,0] is defined but fixed at q⁰[j].
-   z[j,t]: New patients admitted at Hospital j at instant t, where j ∈ H, t ∈ T.
"""
function patient_distribution_model(q⁰, q, u, c)
  n, nt = size(q)
  @assert length(q⁰) == length(u) == n

  model = Model(Cbc.Optimizer)

  @variable(model, x[i=1:n,j=1:n,t=1:nt] ≥ 0, Int)
  @variable(model, 0 ≤ y[j=1:n,t=0:nt] ≤ u[j], Int)
  @variable(model, z[j=1:n,t=1:nt] ≥ 0, Int)

  # Later change to fix
  @constraint(model, [j=1:n], y[j,0] == q⁰[j])

  @objective(model, Min, sum(c[i,j] * x[i,j,t] for i = 1:n, j = 1:n, t = 1:nt))

  # Todos movidos para j são alocados em j
  @constraint(model, [j=1:n,t=1:nt], z[j,t] == sum(x[i,j,t] for i = 1:n))

  # Todos pacientes novos são movidos de i
  @constraint(model, [i=1:n,t=1:nt], q[i,t] == sum(x[i,j,t] for j = 1:n))

  # Todos pacientes em j são atualizados
  @constraint(model, [j=1:n,t=1:nt], y[j,t] == y[j,t-1] + z[j,t])

  optimize!(model)

  x = round.(Int, value.(x))
  y = round.(Int, value.(y))
  z = round.(Int, value.(z))

  return x, y, z, objective_value(model), termination_status(model)
end