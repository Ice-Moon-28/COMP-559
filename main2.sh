# python main.py --func train_gcn --train_dataset mnist --valid_dataset mnist --train_model_dir seed42 --valid_model_dir seed42 --seed 42 --save_path 'gcn/42_mnist_model.pt' > seed42_2.log
python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed42 --valid_dataset mnist --valid_model_dir seed42 --seed 42 --load_path 'gcn/42_mnist_model.pt' 
python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed42 --valid_dataset mnist --valid_model_dir seed888 --seed 42 --load_path 'gcn/42_mnist_model.pt' 
python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed42 --valid_dataset mnist --valid_model_dir seed1228 --seed 42 --load_path 'gcn/42_mnist_model.pt' 
python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed42 --valid_dataset mnist --valid_model_dir seed3047 --seed 42 --load_path 'gcn/42_mnist_model.pt' 


# python main.py --func train_gcn --train_dataset mnist --valid_dataset mnist --train_model_dir seed888 --valid_model_dir seed888 --seed 888 --save_path 'gcn/888_mnist_model.pt' > seed888_2.log
python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed888 --valid_dataset mnist --valid_model_dir seed42 --seed 888 --load_path 'gcn/888_mnist_model.pt' 
python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed888 --valid_dataset mnist --valid_model_dir seed1228 --seed 888 --load_path 'gcn/888_mnist_model.pt' 
python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed888 --valid_dataset mnist --valid_model_dir seed3047 --seed 888 --load_path 'gcn/888_mnist_model.pt' 


# python main.py --func train_gcn --train_dataset mnist --valid_dataset mnist --train_model_dir seed1228 --valid_model_dir seed1228 --seed 1228 --save_path 'gcn/1228_mnist_model.pt' > seed1228_2.log
python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed1228 --valid_dataset mnist --valid_model_dir seed42 --seed 1228 --load_path 'gcn/1228_mnist_model.pt' 
python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed1228 --valid_dataset mnist --valid_model_dir seed888 --seed 1228 --load_path 'gcn/1228_mnist_model.pt' 
python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed1228 --valid_dataset mnist --valid_model_dir seed3047 --seed 1228 --load_path 'gcn/1228_mnist_model.pt' 


# python main.py --func train_gcn --train_dataset mnist --valid_dataset mnist --train_model_dir seed3047 --valid_model_dir seed3047 --seed 3047 --save_path 'gcn/3047_mnist_model.pt' > seed3047_2.log
python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed3047 --valid_dataset mnist --valid_model_dir seed42 --seed 3047 --load_path 'gcn/3047_mnist_model.pt' 
python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed3047 --valid_dataset mnist --valid_model_dir seed888 --seed 3047 --load_path 'gcn/3047_mnist_model.pt' 
python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed3047 --valid_dataset mnist --valid_model_dir seed1228 --seed 3047 --load_path 'gcn/3047_mnist_model.pt' 

