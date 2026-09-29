import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cv2

for name in ["s_after", "00_start", "w_after", "w+d_before", "a_after"]:
    frame = cv2.cvtColor(cv2.imread(f"screenshots/{name}.png"), cv2.COLOR_BGR2RGB)
    middle = frame[250:350, 100:860]            # the area where "YOU LOSE" appears
    white = (middle > 200).all(axis=2)          # pixels where red, green and blue are all bright
    print(name, white.sum())                    # how many white pixels
