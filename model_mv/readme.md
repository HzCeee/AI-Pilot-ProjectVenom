exp1:
just to test the correctness of the new environment
not check the area_ratio

exp2:
check area
add one layer

exp3:
check area
initialization
encode velocity

exp4:
same config as exp3
target speed = 15m/s

exp5:
add target orientation
same config as exp3, except initializing at N(7,5)

exp6:
same config as exp5
initialize orintation

exp7:
same config as exp6
distance: N(5, 5)

exp8:
add target orientation
add hunter velocity
distance: N(5, 5)
no initialization on orieatation

exp9:
same config as exp5
initialize at N(5,5)

it seems like
add target orientation can decrease SR.
initialize orientation will decrease SR. Maybe velocity should be ramdomly initialized as well. 
initialized distance have no effect on SR.
