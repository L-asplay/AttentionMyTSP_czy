#! parameters set
import math

IoT_devices = 30
Rc = 40.0  # coverage / bias
UAV_p = 50

# UAV fly
height = 10
g = 9.8  # gravity
speed = 15
quantity_uav = 2
Cd = 0.3
A = 0.1
air_density = 1.225

P_fly = air_density * A * Cd * pow(speed, 3) / 2 + quantity_uav * g * speed
P_stay = pow(speed, 3)

# Iot device energy compute
switched_capacitance = 1e-6
v = 4

# transmit
B = 1e6
g0 = 20
G0 = 5
upload_P = 3
noise_P = -90
hm = 0
d_2 = pow(Rc, 2) + pow(height, 2)
upload_speed = B * math.log2(1 + g0 * G0 * upload_P / pow(noise_P, 2) / (pow(hm, 2) + d_2))

def getloss(data,p):
    task_data = data["task_data"]
    IoT_resource = data["IoT_resource"]
    UAV_resource = data["UAV_resource"]
    CPU_circles = data["CPU_circles"]

    upl_time = task_data / upload_speed
    UAV_upl_energy = upl_time * UAV_p
    UAV_exe_time = CPU_circles / UAV_resource
    UAV_exe_energy = UAV_exe_time * UAV_p
    IoT_exe_time = CPU_circles / IoT_resource
    IoT_exe_energy = switched_capacitance * pow(IoT_resource, v - 1) * CPU_circles

    upl_time = [0] + sum(upl_time.cpu().numpy().tolist(), [])
    UAV_upl_energy = [0] + sum(UAV_upl_energy.cpu().numpy().tolist(), [])
    UAV_exe_time = [0] + sum(UAV_exe_time.cpu().numpy().tolist(), [])
    UAV_exe_energy = [0] + sum(UAV_exe_energy.cpu().numpy().tolist(), [])
    IoT_exe_time = [0] + sum(IoT_exe_time.cpu().numpy().tolist(), [])
    IoT_exe_energy = [0] + sum(IoT_exe_energy.cpu().numpy().tolist(), [])

    p = p + [0]
    locs = data['task_position'].cpu().numpy().tolist()
    depot = data['UAV_start_pos'].cpu().numpy().tolist()
    loc = depot + locs
  
    dependency = (data["dependencys"]+1).cpu().numpy().tolist()
    prec = [0]*len(loc)
    for i in range(1,len(dependency)):
        prec[dependency[i]]=dependency[i-1]

    time_window = [[0.0,0.0]] + data["time_window"].cpu().numpy().tolist()
    
    fly_energy = 0
    loc_energy = 0
    mec_energy = 0
    cur_l = depot[0]
    cur_t = 0

    for i in range(len(p)):
        mec_energy = mec_energy + UAV_upl_energy[p[i]] + UAV_exe_energy[p[i]] 
        
        fly_t = (math.sqrt((cur_l[0] - loc[p[i]][0] ) ** 2 + (cur_l[1] - loc[p[i]][1]) ** 2) )/speed 
        left = max( max(cur_t + fly_t , time_window[prec[p[i]]][1]), time_window[p[i]][0])
        wait_t = left - cur_t - fly_t

        fly_energy = fly_energy + fly_t*P_fly + wait_t*P_stay
        
        time_window[p[i]][0] = left
        cur_t = left + upl_time[p[i]] + UAV_exe_time[p[i]] 
        if cur_t <= time_window[p[i]][1]:
          time_window[p[i]][1] = cur_t
        elif p[i]>0 :
          return math.inf
        cur_l = loc[p[i]]
    return fly_energy + loc_energy + mec_energy