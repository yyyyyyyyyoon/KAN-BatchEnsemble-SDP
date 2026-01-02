#def count_parameters(model):
#    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import os

mlp_dir = "mlp_results"
for fname in os.listdir(mlp_dir):
    fpath = os.path.join(mlp_dir, fname)
    print(f"{fname}: {os.path.getsize(fpath)} bytes")

