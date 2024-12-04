用于记录对论文  ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS 中代码的修改，  
基于 TSP 代码的无人机边缘计算路由

### usage
```
cd attention-learn-to-route-master/
```
```
python run.py --graph_size 30 --priority 15 2 18 10 --baseline rollout --run_name 'try' --sub_encode_layers 1 --epoch_size 12800 --select_size 25
```
```
python run.py --graph_size 30 --priority 15 2 18 10 21 4 8 17 27 --baseline rollout --run_name 'try' --sub_encode_layers 1 --epoch_size 12800
```
### Tuning hyperparameters
demo_6
```
python run.py --problem 'mec' --graph_size 6 --batch_size 5 --epoch_size 10 --val_size 5 --model 'attention' --sub_encode_layers 1 --select_size 6 --priority 3 5 1 --n_epochs 10 --eval_batch_size 5 --baseline rollout --run_name 'demo_6' 
```