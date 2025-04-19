import torch
import torch.nn.functional as F

from dataset.dataset import get_dataset
from gnn.model.gcn import EdgeGCNRegressor
from util.graph import extract_neuron_level_graph_without_input_layer
from util.model import set_seed

def compute_tensor_accuracy(pred: torch.Tensor, target: torch.Tensor, atol: float = 1e-5) -> float:
    """
    Computes the accuracy between two tensors by counting the proportion of elements
    that are close within a specified absolute tolerance.

    Args:
        pred (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Target tensor.
        atol (float, optional): Absolute tolerance. Defaults to 1e-5.

    Returns:
        float: Accuracy as the proportion of matching elements.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred shape {pred.shape} vs target shape {target.shape}")

    # Determine which elements are close within the specified tolerance
    matching_elements = torch.isclose(pred, target, atol=atol)

    # Calculate the number of matching elements
    num_matches = matching_elements.sum().item()

    # Calculate the total number of elements
    total_elements = pred.numel()

    # Compute accuracy
    accuracy = num_matches / total_elements

    return accuracy

def get_model_path(model_dir, model_name):
    if model_name == 'mnist':
        return "{}/mnist_model.pt".format(model_dir)
    elif model_name == 'fashion_mnist':
        return "{}/fashion_mnist_model.pt".format(model_dir)
    elif model_name == 'text_classification':
        return "{}/text_classification_model.pt".format(model_dir)
    elif model_name == 'imdb_classification':
        return "{}/imdb_classification_model.pt".format(model_dir)
    else:
        raise NotImplementedError

def eval_gnn(
        model,
        test_data, 
        loss_fn=F.mse_loss,
        load_path=None,
        device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    ):

    model.to(device)
    model.eval()

    if load_path:
        model.load_state_dict(torch.load(load_path, map_location=device))
        print(f"📥 Loaded model from {load_path}")
    else:
        print("⚠️ No model path provided. Evaluating with current model state.")

    with torch.no_grad():
        x = model(test_data.x.to(device),
                  test_data.edge_index.to(device),
                  test_data.edge_attr.to(device))
        
        pred = model.predict(x=x, edge_index=test_data.edge_index.to(device))
        target = test_data.edge_attr.squeeze().to(device)

        if torch.isnan(pred).any() or torch.isnan(target).any():
            print("⚠️ NaN detected in predictions or targets during evaluation!")
            return

        loss = loss_fn(pred, target)
        mae = F.l1_loss(pred, target)

        print(f"🧪 Eval Loss (MSE): {loss.item():.6f}")
        print(f"📊 Eval MAE: {mae.item():.6f}")

    return {
        "mse": loss.item(),
        "mae": mae.item(),
        "pred": pred.cpu(),
        "target": target.cpu()
    }

def eval_gcn(
    train_dataset_name='mixer_b16',
    test_dataset_name='mixer_b16',   
    seed=3047,
    load_path=None,
    train_dataset_dir=None,
    test_dataset_dir=None,
):
    
    set_seed(seed=seed)

    model = EdgeGCNRegressor(in_channels=5, hidden_channels=128, out_channels=64)

    test_mlp_model = get_dataset(test_dataset_name, path=get_model_path(test_dataset_dir, test_dataset_name))

    # train_mlp_model = get_dataset(train_dataset_name, path=get_model_path(train_dataset_dir, train_dataset_name))

    num_linear_layers = sum(1 for layer in test_mlp_model.modules() if isinstance(layer, torch.nn.Linear))
    print(f"Number of Linear layers: {num_linear_layers}")

    test_data = extract_neuron_level_graph_without_input_layer(test_mlp_model, num_linear_layers)

    result = eval_gnn(
        model=model,
        test_data=test_data,
        load_path=load_path,
    )

    # 把 edge_attr 拿出来并转到相同设备
    pred = result["pred"]
    target = result['target']

    # 确保形状一致
    assert pred.shape == target.shape, "Edge attribute shapes do not match!"

    # 计算 MSE Loss
    mse_loss = F.mse_loss(pred, target)
    mae_loss = F.l1_loss(pred, target)

    print(f"🔍 MSE Loss between edge_attr of test and train: {mse_loss.item():.6f}")
    print(f"📊 MAE Loss between edge_attr of test and train: {mae_loss.item():.6f}")

    mse_loss_zero = F.mse_loss(pred, torch.zeros_like(pred))
    mae_loss_zero = F.l1_loss(pred, torch.zeros_like(pred))

    print(f"🔍 MSE Loss between edge_attr of test and zero: {mse_loss_zero.item():.6f}")
    print(f"📊 MAE Loss between edge_attr of test and zero: {mae_loss_zero.item():.6f}")

    print("📊 Accuracy between edge_attr of test and train:", compute_tensor_accuracy(pred, target, 1e-1))

    print("📊 Accuracy between edge_attr of test and train:", compute_tensor_accuracy(pred, target, 5e-2))

    print("📊 Accuracy between edge_attr of test and train:", compute_tensor_accuracy(pred, target, 1e-2))

    print("📊 Accuracy between edge_attr of test and train:", compute_tensor_accuracy(pred, target, 5e-3))

    ### Accuracy between edge_attr of test and train: 0.9993942011884193

    ### Accuracy between edge_attr of test and train: 0.9621263801677272

    ### Accuracy between edge_attr of test and train: 0.26842286864174636

    ### Accuracy between edge_attr of test and train: 0.13452091849635467

    ### Accuracy between edge_attr of test and train: 0.02711674006489949

    print("📊 Accuracy between edge_attr of test and train:", compute_tensor_accuracy(pred, target, 1e-3))

    print("📊 Accuracy between edge_attr of test and train:", compute_tensor_accuracy(pred, target, 1e-4))

    print(
        "x shape:", test_data.x.shape,
        "edge_index shape:", test_data.edge_index.shape,
        "edge_attr shape:", test_data.edge_attr.shape
    )

    


def eval_gan(
    train_dataset_name='mixer_b16',
    test_dataset_name='mixer_b16',   
    seed=3047,
    load_path=None,
    train_dataset_dir=None,
    test_dataset_dir=None,
):
    
    set_seed(seed=seed)

    model = EdgeGCNRegressor(in_channels=5, hidden_channels=128, out_channels=64)

    test_mlp_model = get_dataset(test_dataset_name, path=get_model_path(test_dataset_dir, test_dataset_name))

    train_mlp_model = get_dataset(train_dataset_name, path=get_model_path(train_dataset_dir, train_dataset_name))

    num_linear_layers = sum(1 for layer in test_mlp_model.modules() if isinstance(layer, torch.nn.Linear))
    print(f"Number of Linear layers: {num_linear_layers}")

    test_data = extract_neuron_level_graph_without_input_layer(test_mlp_model, num_linear_layers)

    train_data = extract_neuron_level_graph_without_input_layer(train_mlp_model, num_linear_layers)

    # 把 edge_attr 拿出来并转到相同设备
    pred = test_data.edge_attr.squeeze()
    target = train_data.edge_attr.squeeze()

    import pdb; pdb;pdb.set_trace()

    # 确保形状一致
    assert pred.shape == target.shape, "Edge attribute shapes do not match!"

    # 计算 MSE Loss
    mse_loss = F.mse_loss(pred, target)
    mae_loss = F.l1_loss(pred, target)



    print(f"🔍 MSE Loss between edge_attr of test and train: {mse_loss.item():.6f}")
    print(f"📊 MAE Loss between edge_attr of test and train: {mae_loss.item():.6f}")

    mse_loss_zero = F.mse_loss(pred, torch.zeros_like(pred))
    mae_loss_zero = F.l1_loss(pred, torch.zeros_like(pred))

    print(f"🔍 MSE Loss between edge_attr of test and zero: {mse_loss_zero.item():.6f}")
    print(f"📊 MAE Loss between edge_attr of test and zero: {mae_loss_zero.item():.6f}")

    print(
        "x shape:", test_data.x.shape,
        "edge_index shape:", test_data.edge_index.shape,
        "edge_attr shape:", test_data.edge_attr.shape
    )

    eval_gnn(
        model=model,
        test_data=test_data,
        load_path=load_path,
    )