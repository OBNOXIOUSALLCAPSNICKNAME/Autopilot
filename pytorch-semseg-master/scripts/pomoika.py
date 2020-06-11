import  numpy as np
import math
import cv2

#p1_1 = (2, 650)
#p2_1 = (1012, 806)

p1_1 = (100, 200)
p2_1 = (800, 200)


p1_2 = (1436, 525)
p2_2 = (190, 805)

slope_1 = np.arctan2((p1_1[1] - p2_1[1]), (p1_1[0] - p2_1[0]))
slope_2 = np.arctan2((p1_2[1] - p2_2[1]), (p1_2[0] - p2_2[0]))

slope_1_deg = np.rad2deg(slope_1)
slope_2_deg = np.rad2deg(slope_2)

print(slope_1_deg)
print(slope_2_deg)
