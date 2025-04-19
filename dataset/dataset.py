from dataset.pre_trained.mixer_b16 import get_mixer_b16
from dataset.simple_task.simple import get_imdb_classification_model, get_mnist_model, get_fashion_mnist_model, get_text_classification_model


def get_dataset(dataset_name, path=None):
    if dataset_name == 'mnist':
        return get_mnist_model(path=path)
    elif dataset_name == 'fashion_mnist':
        return get_fashion_mnist_model(path=path)
    elif dataset_name == 'text_classification':
        return get_text_classification_model(path=path)
    
    elif dataset_name == 'imdb_classification':
        return get_imdb_classification_model(path=path)
    elif dataset_name == 'mixer_b16':
        return get_mixer_b16(path=path)
    else:
        raise NotImplementedError