
python main.py --func train_gcn --train_dataset mnist --valid_dataset mnist --train_model_dir seed3047 --valid_model_dir seed3047 --seed 3047 --save_path 'gcn/mnist_model.pt' 
python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed3047 --valid_dataset mnist --valid_model_dir seed42 --seed 3047 --load_path 'gcn/mnist_model.pt' 
python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed3047 --valid_dataset mnist --valid_model_dir seed888 --seed 3047 --load_path 'gcn/mnist_model.pt' 
python main.py --func eval_gcn --train_dataset mnist  --train_model_dir seed3047 --valid_dataset mnist --valid_model_dir seed1228 --seed 3047 --load_path 'gcn/mnist_model.pt' 

