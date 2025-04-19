import torch
from dataset.dataset import get_dataset
from dataset.pre_trained.mixer_b16 import get_mixer_b16
from gnn.model.gan import EdgeGATRegressor
from gnn.model.gcn import EdgeGCNRegressor
from gnn.model.graph_sage import EdgeGraphSAGERegressor
from gnn.model.pna import EdgePNARegressor
from gnn.train_pipeline import train_gnn
from util.graph import extract_neuron_level_graph_with_input_layer, extract_neuron_level_graph_without_input_layer, split_edges
from util.model import set_seed


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


def train_gcn(
        train_dataset_name='mixer_b16',   
        valid_dataset_name='mixer_b16',
        seed=3047,
        save_path=None,
        train_model_dir=None,
        valid_model_dir=None,
    ):


    set_seed(seed=seed)

    model = EdgeGCNRegressor(in_channels=5, hidden_channels=128, out_channels=64)

    train_mlp_model = get_dataset(train_dataset_name, path=get_model_path(train_model_dir, train_dataset_name))

    valid_mlp_model = get_dataset(valid_dataset_name, path=get_model_path(valid_model_dir, valid_dataset_name))

    num_linear_layers = sum(1 for layer in train_mlp_model.modules() if isinstance(layer, torch.nn.Linear))
    print(f"Number of Linear layers: {num_linear_layers}")

    train_data = extract_neuron_level_graph_without_input_layer(train_mlp_model, num_linear_layers)

    valid_data = extract_neuron_level_graph_without_input_layer(valid_mlp_model, num_linear_layers)

    print(
        "x shape:", train_data.x.shape,
        "edge_index shape:", train_data.edge_index.shape,
        "edge_attr shape:", train_data.edge_attr.shape
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    train_gnn(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        optimizer=optimizer,
        epochs=250,
        early_stop_patience=25,
        device='mps',
        save_path=save_path,
    )


def train_gan(
        dataset_name='mixer_b16',    
    ):

    set_seed(3047)

    model = EdgeGATRegressor(
        in_channels=5,
        hidden_channels=64,
        out_channels=64,
        linear_hidden_channels=128,
        heads=8,
    )

    mlp_model = get_dataset(dataset_name)
    num_linear_layers = sum(1 for layer in mlp_model.modules() if isinstance(layer, torch.nn.Linear))
    print(f"Number of Linear layers: {num_linear_layers}")

    Data = extract_neuron_level_graph_without_input_layer(mlp_model, num_linear_layers)

    print(
        "x shape:", Data.x.shape,
        "edge_index shape:", Data.edge_index.shape,
        "edge_attr shape:", Data.edge_attr.shape
    )

    train_data, valid_data = split_edges(Data.x, Data.edge_index, Data.edge_attr, split_ratio=0.8, seed=42)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    train_gnn(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        optimizer=optimizer,
        epochs=250,
        device='mps'
    )

def train_graph_sage(
        dataset_name='mixer_b16',    
    ):

    set_seed(3047)

    model = EdgeGraphSAGERegressor(
        in_channels=6,
        hidden_channels=64,
        out_channels=64,
    )

    mlp_model = get_dataset(dataset_name)
    num_linear_layers = sum(1 for layer in mlp_model.modules() if isinstance(layer, torch.nn.Linear))
    print(f"Number of Linear layers: {num_linear_layers}")

    Data = extract_neuron_level_graph_without_input_layer(mlp_model, num_linear_layers)

    print(
        "x shape:", Data.x.shape,
        "edge_index shape:", Data.edge_index.shape,
        "edge_attr shape:", Data.edge_attr.shape
    )

    train_data, valid_data = split_edges(Data.x, Data.edge_index, Data.edge_attr, split_ratio=0.8, seed=42)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_gnn(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        optimizer=optimizer,
        epochs=1000,
        device='cpu'
    )



def train_graph_pna(
        dataset_name='mixer_b16',    
    ):

    set_seed(3047)

    model = EdgePNARegressor(
        in_channels=6,
        hidden_channels=64,
        out_channels=64,
    )

    mlp_model = get_dataset(dataset_name)
    num_linear_layers = sum(1 for layer in mlp_model.modules() if isinstance(layer, torch.nn.Linear))
    print(f"Number of Linear layers: {num_linear_layers}")

    Data = extract_neuron_level_graph_without_input_layer(mlp_model, num_linear_layers)

    print(
        "x shape:", Data.x.shape,
        "edge_index shape:", Data.edge_index.shape,
        "edge_attr shape:", Data.edge_attr.shape
    )

    train_data, valid_data = split_edges(Data.x, Data.edge_index, Data.edge_attr, split_ratio=0.8, seed=42)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_gnn(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        optimizer=optimizer,
        epochs=1000,
        device='cpu'
    )


    