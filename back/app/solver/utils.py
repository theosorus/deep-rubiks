import random

import torch


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def save_model(artifacts, filepath: str):
    torch.save({
        'model_state_dict': artifacts.net.state_dict(),
        'target_model_state_dict': artifacts.target_net.state_dict(),
        'config': artifacts.cfg,
        'history': artifacts.history
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str, device=None):
    from solver.cost_to_go_net import CostToGoNet
    from solver.artifacts import DaviArtifacts
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(filepath, map_location=device,weights_only=False)
    
    cfg = checkpoint['config']
    cfg.device = device
    
    net = CostToGoNet(
        cfg.input_dim,
        hidden1=cfg.hidden1,
        hidden2=cfg.hidden2,
        num_res_blocks=cfg.num_res_blocks
    ).to(device)
    
    target_net = CostToGoNet(
        cfg.input_dim,
        hidden1=cfg.hidden1,
        hidden2=cfg.hidden2,
        num_res_blocks=cfg.num_res_blocks
    ).to(device)
    
    net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['target_model_state_dict'])
    
    artifacts = DaviArtifacts(
        net=net,
        target_net=target_net,
        history=checkpoint['history'],
        cfg=cfg
    )
    
    print(f"Model loaded from {filepath}")
    return artifacts


def init_solver(model_path: str, device: torch.device):
    from solver.utils import load_model, get_device
    from solver.rubiks_cube_adapter import RubiksCubeAdapter
    from solver.astar import AStarSolver
    
    artifacts = load_model(model_path, device=device)
    adapter = RubiksCubeAdapter()
    solver = AStarSolver(artifacts.net, adapter)
    
    return solver
    