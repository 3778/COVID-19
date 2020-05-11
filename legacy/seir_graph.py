import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import matplotlib.pyplot as plt
import pandas as pd

# initial conditions and params.
S0 = 20_000_000
I0 = 250
E0 = 0
R0 = 1

# Network topology
g = nx.erdos_renyi_graph(S0, 0.1)

# Model selection
model = ep.SEIRModel(g)

# Model Configuration
cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.01)
cfg.add_model_parameter('gamma', 0.005)
cfg.add_model_parameter('alpha', 0.05)
cfg.add_model_parameter('fraction_infected', 0.05)
model.set_initial_status(cfg)

# Simulation execution
iterations = model.iteration_bunch(200)
df = (pd.DataFrame([i['node_count'] for i in iterations])
        .rename(columns={0: 'Susceptible',
                         1: 'Infected',
                         2: 'Exposed',
                         3: 'Removed'}))
