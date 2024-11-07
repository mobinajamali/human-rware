import torch as T

state_dict = T.load('./results/models/mappo_seed663242369_rware:rware-tiny-4ag-v2_2024-11-02 16:24:42.284341/5000500/agent.th')
print(f"input_dim: {state_dict['fc1.weight'].T.shape[0]}")