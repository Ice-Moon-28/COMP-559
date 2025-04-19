from dataset.simple_task.simple import main as train_simple_task_model

import argparse

from gnn.model.eval import eval_gan, eval_gcn
from gnn.model.train import train_gan, train_gcn, train_graph_pna, train_graph_sage

def main():
    parser = argparse.ArgumentParser(description="Run specific task")
    parser.add_argument('--func', type=str, default='train_simple_task_model',
                        help='Which task to run')
    
    parser.add_argument('--train_dataset', type=str, default='mnist',
                        help='Which task to run')
    
    parser.add_argument('--valid_dataset', type=str, default='mnist',
                        help='Which task to run')
    
    parser.add_argument('--train_model_dir', type=str, default='.')

    parser.add_argument('--valid_model_dir', type=str, default='.')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    
    parser.add_argument('--dir', type=str, default='model',
                        help='model save directory')
    
    parser.add_argument('--save_path', type=str, default='.',
                        help='save_model_path')
    
    parser.add_argument('--load_path', type=str, default='.',
                        help='save_model_path')

    args = parser.parse_args()

    if args.func == 'train_simple_task_model':
        train_simple_task_model(
            seed=args.seed,
            dir_name=args.dir
        )

    elif args.func == 'train_gcn':
        train_gcn(
            train_dataset_name=args.train_dataset,
            seed=args.seed,
            valid_dataset_name=args.valid_dataset,
            train_model_dir=args.train_model_dir,
            valid_model_dir=args.valid_model_dir,
            save_path=args.save_path
        )

    elif args.func == 'train_gan':
        train_gan(
            dataset_name=args.dataset
        )

    elif args.func == 'train_graph_sage':
        train_graph_sage(
            dataset_name=args.dataset
        )

    elif args.func == 'train_pna':
        train_graph_pna(
            dataset_name=args.dataset
        )

    elif args.func == 'eval_gcn':
        eval_gcn(
            train_dataset_name=args.train_dataset,
            test_dataset_name=args.valid_dataset,
            seed=args.seed,
            load_path=args.load_path,
            train_dataset_dir=args.train_model_dir,
            test_dataset_dir=args.valid_model_dir
        )
    
    elif args.func == 'eval_gan':
        eval_gan(
            train_dataset_name=args.train_dataset,
            test_dataset_name=args.valid_dataset,
            seed=args.seed,
            load_path=args.load_path,
            train_dataset_dir=args.train_model_dir,
            test_dataset_dir=args.valid_model_dir
        )

    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()