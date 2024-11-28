recode of changes : on new problem
<font  size =1>no 缩放矩阵</font>

### target 
    想研究的问题是：
    地面端有很多数据处理的任务，任务之间有先后顺序关系，
    地面端处理能力有限，对处理速度有要求的任务,需要加载到带有服务器的无人机端处理，
    无人机首先选择其要加载过来的地面端任务，选择完之后
    就是之前研究的带有顺序约束的tsp问题,只不过是优化目标变成无人机的总能耗，
    每个任务有时间窗口，无人机应该在比时间窗口完成处理。

    需要改代码的几个部分：
    首先是数据生成部分，
    其次是无人机站选择哪些地面端任务处理的网络结构,
    解码部分,
    要加时间窗口，改优化目标, mask激励。

## 无人机选择任务的网络结构代码 
### task_selection.py/class NodeSelector(nn.Module)
### task_selection.py/class AttNodeSelector(nn.Module)

##  无人机数据生成     
### generate_data.py/generate_uav_data(dataset_size, uav_size, dependency=[])

##  任务 encoder 和无人机信息编码
### attention_model.py

##  decoder 
### attention_model.py/_inner
#### attention_model.py/_get_log_p
#### attention_model.py/_get_parallel_step_context
#### problems/mec/state_mec.py
#### problems/mec/problem_mec.py


## changed & to be realize/check
{
problem_mec.py/get_cost
eval.py/_eval_dataset
}



输入解包x
{

  "task_position": (batch_size, gena_size, 2),   
  "time_window": (batch_size, gena_size, 2),   
  "loT_resource": (batch_size,gena_size,1),  
  "UAV_start_pos": (batch_size, 1, 2),   
  "UAV_resource": (batch_size, 1, 1)  
}

{
python run.py --graph_size 30 --priority 2 1 3 --baseline rollout --run_name 'try' --sub_encode_layers 1 --epoch_size 12800
}
