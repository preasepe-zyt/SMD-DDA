# SMD-DDA
<h1> Environment Setup<br> 
<h1> Pretrained module-Pre_Gene <br> 
<h4> conda env create -f gene_train/gene.yml <br> 
<h1> Pretrained module-Pre_Cliff  <br> 
<h4> conda env create -f cliff_train/cliff_env.yml <br> 
<h1> Pretrained module-Pre_Property <br> 
<h4> conda env create -f property_train/property_env.yml <br>
<h1> Main Task <br> 
<h4> conda env create -f SMD-DDA.yml<br> 
<h1> Pretrained module Training <br>
<h4> python train_gene.py <br>
<h4> python main.py --dataset 'HEMBL239_EC50' --data_dir 'Data' --model_dir './checkpoints/'  --loss 'MSE+direction' --sim_threshold 0.9 --dist_threshold 1.0 --epochs 500 --split_method cliff --epochs 500 --num_folds 10 <br>
<h4> python main.py --dataset 'CHEMBL244_Ki' --data_dir 'Data' --model_dir './checkpoints/'  --loss 'MSE+direction' --sim_threshold 0.9 --dist_threshold 1.0 --epochs 500 --split_method cliff --epochs 500 --num_folds 10 <br>
<h4> python main.py --dataset 'BBBP' --data_dir 'Data' --model_dir './checkpoints/'  --loss 'BCE' --epochs 500 --split_method random  --epochs 500 --num_folds 10 --task 'classificfation' --metric auprc <br>
<h4> python main.py --dataset 'Clintox' --data_dir 'Data' --model_dir './checkpoints/' --loss 'BCE' --epochs 500 --split_method random --epochs 500 --num_folds 10 --task 'classificfation' --metric auroc <br>
<h4> python main.py --dataset 'SIDER'  --data_dir 'Data' --model_dir './checkpoints/' --loss 'BCE' --epochs 500 --split_method 'random'  --epochs 500 --num_folds 10 --task 'classificfation' --metric auprc <br>
<h1> Main Task Training <br>
<h4> python demo.py <br>
<h1> Illustration of the SMD-DDA framework. <br>
<img src="framework.jpg" alt=""/>

