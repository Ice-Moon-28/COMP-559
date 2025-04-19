# python main.py --func train_simple_task_model --seed 42 --dir seed42 > seed42.log
# python main.py --func train_simple_task_model --seed 3047 --dir seed3047 > seed3047.log
# python main.py --func train_simple_task_model --seed 888 --dir seed888 > seed888.log
# python main.py --func train_simple_task_model --seed 1228 --dir seed1228 > seed1228.log

# python main.py --func train_gcn --train_dataset mnist  --train_model_dir seed42 --valid_dataset fashion_mnist --valid_model_dir seed42 --seed 42 --load_path 'gcn/42_mnist_model.pt' 

# python main.py --func train_gcn --train_dataset mnist  --train_model_dir seed42 --valid_dataset mnist --valid_model_dir seed42 --seed 42 --load_path 'gcn/42_mnist_model.pt' 

# python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed42 --valid_dataset fashion_mnist --valid_model_dir seed42 --seed 42 --load_path 'gcn/42_mnist_model.pt' 
# python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed42 --valid_dataset fashion_mnist --valid_model_dir seed888 --seed 42 --load_path 'gcn/42_mnist_model.pt' 
# python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed42 --valid_dataset fashion_mnist --valid_model_dir seed1228 --seed 42 --load_path 'gcn/42_mnist_model.pt' 
# python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed42 --valid_dataset fashion_mnist --valid_model_dir seed3047 --seed 42 --load_path 'gcn/42_mnist_model.pt' 

# python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed3047 --valid_dataset text_classification --valid_model_dir seed42 --seed 3047 --load_path 'gcn/42_mnist_model.pt' 
# python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed3047 --valid_dataset text_classification --valid_model_dir seed888 --seed 3047 --load_path 'gcn/42_mnist_model.pt' 
# python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed3047 --valid_dataset text_classification --valid_model_dir seed1228 --seed 3047 --load_path 'gcn/42_mnist_model.pt' 


# python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed1228 --valid_dataset imdb_classification --valid_model_dir seed42 --seed 1228 --load_path 'gcn/42_mnist_model.pt' 
# python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed1228 --valid_dataset imdb_classification --valid_model_dir seed888 --seed 1228 --load_path 'gcn/42_mnist_model.pt'  
# python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed1228 --valid_dataset imdb_classification --valid_model_dir seed3047 --seed 1228 --load_path 'gcn/42_mnist_model.pt' 


