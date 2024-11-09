recode of changes : on  state 2
<font  size =1>no 缩放矩阵</font>

### target 
    想研究的问题是：
    地面端有很多数据处理的任务，任务之间有先后顺序关系，
    地面端处理能力有限，对处理速度有要求的任务,需要加载到带有服务器的无人机端处理，
    无人机首先选择其要加载过来的地面端任务，选择完之后
    就是之前研究的带有顺序约束的tsp问题,只不过是优化目标变成无人机的总能耗，
    每个任务有时间窗口，无人机应该在比时间窗口完成处理。
    需要改代码的几个部分：
    首先是数据生成部分，这部分可以等同学写好再改；
    其次是无人机站选择哪些地面端任务处理，这个需要加个网络结构,
    先加个简单的全链接层试试,或者也利用注意力机制选择,
    最后之前写的代码，要加时间窗口，改优化目标, mask激励。

## 无人机选择任务的网络结构代码 

### task_selection.py
     
### generate_data.py/generate_uav_data(dataset_size, uav_size, dependency=[])
```py
def generate_uav_data(dataset_size, uav_size, dependency=[]):

    return list(zip(
        task_data.tolist(),
        UAV_start_pos.tolist(),
        task_position.tolist(),
        IoT_resource.tolist(),
        UAV_resource.tolist(),
        time_window.tolist()
    ))
    
```

{
AttentionModel:
  loc_embeding
  _get_parallel_step_context

eval:
  _eval_dataset
}



输入解包x
{

  "task_position": (batch_size, gena_size, 2),  
  "time_window": (batch_size, gena_size, 2),  
  "loT_resource": (batch_size,gena_size,1)
  "UAV_start_pos": (batch_size, 1, 2) 
  "UAV_resource": (batch_size, 1, 1)
}

