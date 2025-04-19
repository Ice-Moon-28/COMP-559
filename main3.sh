# python main.py --func train_simple_task_model --seed 42 --dir seed42 > seed42.log
# python main.py --func train_simple_task_model --seed 3047 --dir seed3047 > seed3047.log
# python main.py --func train_simple_task_model --seed 888 --dir seed888 > seed888.log
# python main.py --func train_simple_task_model --seed 1228 --dir seed1228 > seed1228.log

# python main.py --func train_gcn --train_dataset imdb_classification --valid_dataset imdb_classification --train_model_dir seed42 --valid_model_dir seed42 --seed 42 --save_path 'gcn/42_imdb_classification_model.pt' > seed42_3.log
python main.py --func eval_gcn --train_dataset imdb_classification  --train_model_dir seed42 --valid_dataset imdb_classification --valid_model_dir seed888 --seed 42 --load_path 'gcn/42_imdb_classification_model.pt' 
python main.py --func eval_gcn --train_dataset imdb_classification  --train_model_dir seed42 --valid_dataset imdb_classification --valid_model_dir seed1228 --seed 42 --load_path 'gcn/42_imdb_classification_model.pt' 
python main.py --func eval_gcn --train_dataset imdb_classification  --train_model_dir seed42 --valid_dataset imdb_classification --valid_model_dir seed3047 --seed 42 --load_path 'gcn/42_imdb_classification_model.pt' 


# python main.py --func train_gcn --train_dataset imdb_classification --valid_dataset imdb_classification --train_model_dir seed888 --valid_model_dir seed888 --seed 888 --save_path 'gcn/888_imdb_classification_model.pt' > seed888_3.log
python main.py --func eval_gcn --train_dataset imdb_classification  --train_model_dir seed888 --valid_dataset imdb_classification --valid_model_dir seed42 --seed 888 --load_path 'gcn/888_imdb_classification_model.pt' 
python main.py --func eval_gcn --train_dataset imdb_classification  --train_model_dir seed888 --valid_dataset imdb_classification --valid_model_dir seed888 --seed 888 --load_path 'gcn/888_imdb_classification_model.pt' 
python main.py --func eval_gcn --train_dataset imdb_classification  --train_model_dir seed888 --valid_dataset imdb_classification --valid_model_dir seed1228 --seed 888 --load_path 'gcn/888_imdb_classification_model.pt' 
python main.py --func eval_gcn --train_dataset imdb_classification  --train_model_dir seed888 --valid_dataset imdb_classification --valid_model_dir seed3047 --seed 888 --load_path 'gcn/888_imdb_classification_model.pt' 


python main.py --func train_gcn --train_dataset imdb_classification --valid_dataset imdb_classification --train_model_dir seed1228 --valid_model_dir seed1228 --seed 1228 --save_path 'gcn/1228_imdb_classification_model.pt' > seed1228_3.log
python main.py --func eval_gcn --train_dataset imdb_classification  --train_model_dir seed1228 --valid_dataset imdb_classification --valid_model_dir seed42 --seed 1228 --load_path 'gcn/1228_imdb_classification_model.pt' 
python main.py --func eval_gcn --train_dataset imdb_classification  --train_model_dir seed1228 --valid_dataset imdb_classification --valid_model_dir seed888 --seed 1228 --load_path 'gcn/1228_imdb_classification_model.pt' 
python main.py --func eval_gcn --train_dataset imdb_classification  --train_model_dir seed1228 --valid_dataset imdb_classification --valid_model_dir seed3047 --seed 1228 --load_path 'gcn/1228_imdb_classification_model.pt' 


python main.py --func train_gcn --train_dataset imdb_classification --valid_dataset imdb_classification --train_model_dir seed3047 --valid_model_dir seed3047 --seed 3047 --save_path 'gcn/3047_imdb_classification_model.pt' > seed3047_3.log
python main.py --func eval_gcn --train_dataset imdb_classification  --train_model_dir seed3047 --valid_dataset imdb_classification --valid_model_dir seed42 --seed 3047 --load_path 'gcn/3047_imdb_classification_model.pt' 
python main.py --func eval_gcn --train_dataset imdb_classification  --train_model_dir seed3047 --valid_dataset imdb_classification --valid_model_dir seed888 --seed 3047 --load_path 'gcn/3047_imdb_classification_model.pt' 
python main.py --func eval_gcn --train_dataset imdb_classification  --train_model_dir seed3047 --valid_dataset imdb_classification --valid_model_dir seed1228 --seed 3047 --load_path 'gcn/3047_imdb_classification_model.pt' 

