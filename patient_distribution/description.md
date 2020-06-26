# model.jl - How to use 

- [model.jl - How to use](#modeljl---how-to-use)
  - [Install](#install)
  - [Example using fake data](#example-using-fake-data)
  - [Description](#description)
    - [Model](#model)
    - [Model IO](#model-io)

## Install

Install `julia`. My suggestion is to use [jill](https://github.com/abelsiqueira/jill):

```shell
$ bash -ci "$(curl -fsSL https://raw.githubusercontent.com/abelsiqueira/jill/master/jill.sh)"
```

The model was tested with Julia version `1.4`.

Install the Julia packages:

```shell
$ julia --project=. -e 'using Pkg; pkg"instantiate"'
```

## Example using fake data

```shell
$ julia --project=. model-io.jl
```

## Description

### Model

The file `model.jl` provides a simple model for patient distribution. The following is the docstring for `patient_distribution_model`:

```julia
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
```

### Model IO

- `data.csv` provides fake data that `model-io.jl` reads to call `patient_distribution_model` above.
- `model-io.jl` then produces files `solution-*.csv` with the solution.
  - `x` is stored in `solution-move.csv` with each line as `(t,i,j,x[i,j,t])`.
  - `y` is stored in `solution-occupation.csv`.
  - `z` is stored in `solution-new.csv`.
