#! parameters set
import math

IoT_devices = 30
Rc = 40.0  # coverage / bias
UAV_p = 50

# UAV fly
height = 10
g = 9.8  # gravity
speed = 20
quantity_uav = 2
Cd = 0.3
A = 0.1
air_density = 1.225

P_fly = air_density * A * Cd * pow(speed, 3) / 2 + quantity_uav * g * speed
P_stay = pow(speed, 3)

# Iot device energy compute
switched_capacitance = 1e-7
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
