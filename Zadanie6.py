import cv2
import numpy as np
import json
import os

cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

position = []
    

def on_mouse_click(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global position
        position = [y, x]
        
cv2.setMouseCallback("Camera", on_mouse_click)

measures = []
bgr_color = []
hsv_color = []

cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

colors = {}
if os.path.isfile('calibration_data.json'):
    with open('calibration_data.json') as f_json:       
        colors = json.load(f_json)
else:
    colors = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Purple']
    colors = {i: {'lower': [], 'upper': []} for i in colors}

color_names = iter(colors.keys())
color = next(color_names)
probes = []
calibrated = True
while True:
    
    key = cv2.waitKey(1)
    _, frame = cam.read()
    blurred = cv2.GaussianBlur(frame, (11,11),0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    if not os.path.isfile('calibration_data.json') or not calibrated:
        cv2.putText(frame, f"Please, calibrate colors for successful object recognition", 
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        

        if color:
            cv2.putText(frame, f'Calibrate "{color}" color (click 7 times on an item) or press "s" to skip, or "d" to delete color', 
                        (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            
            if key == ord('s'):
                try:
                    color = next(color_names)
                except StopIteration:
                    color = ""                
            if key == ord('d'):
                colors[color]['upper'].clear()
                colors[color]['lower'].clear()
                try:
                    color = next(color_names)
                except StopIteration:
                    color = "" 
            if position:
                cv2.circle(frame, (position[1], position[0]), 5, 255)
                pixel_color = blurred[position[0], position[1], :]
                measures.append(pixel_color)
                if len(measures) >= 10:
                    bgr_color = np.uint8([[np.average(measures,0)]])
                    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)
                    bgr_color = bgr_color[0][0]
                    hsv_color = hsv_color[0][0]
                    measures.clear()
                    if hsv_color[0] > 174:
                        hsv_color[0] = hsv_color[0] - 180
                    probes += [hsv_color]
                    position.clear()
                    if len(probes) == 7:
                        colors[color]['upper'] = np.max(probes, 0).tolist()
                        colors[color]['lower'] = np.min(probes, 0).tolist() 
                        probes.clear()
                        try:
                            color = next(color_names)
                        except StopIteration:
                            color = ""
                                        
        else:
            for c1 in colors:
                for c2 in colors:
                    if (colors[c1]['upper'] and colors[c2]['upper'] and 
                    colors[c1]['upper'][0] > colors[c2]['upper'][0]):
                        while colors[c1]['lower'][0] <= colors[c2]['upper'][0]:
                            colors[c1]['lower'][0] +=1
            with open('calibration_data.json', 'w') as f_json:  
                json.dump(colors, f_json)
            calibrated = True           
                    
        cv2.putText(frame, f"Color BGR={bgr_color}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(frame, f"Color HSV={hsv_color}", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                     
    else: 
        merge_mask = 0
        balls_pos = {}
        for name, color in colors.items():
            if color['upper'] and color['lower']:
                lower = np.uint8(color['lower'])
                upper = np.uint8(color['upper'])
                mask = cv2.inRange(hsv, lower, upper)
                #mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=8)
                if type(merge_mask) == int:
                    merge_mask = mask
                else:
                    merge_mask += mask
                bgr = cv2.cvtColor(np.uint8([[upper]]), cv2.COLOR_HSV2BGR)[0][0].tolist()                
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                if len(cnts)>0:
                    cnt = max(cnts, key=cv2.contourArea)
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    if radius > 50:
                        balls_pos[name] = (x, y)
                        cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
                        cv2.circle(frame, (int(x), int(y)), 5,(0,255,255), -1)
                        cv2.putText(frame, f"{name} ball", (int(x-len(name)*15), int(y+radius+30)), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr, 2)
        res = " ".join(sorted(balls_pos, key=lambda p: balls_pos[p][0]))
        balls_pos.clear()              
        cv2.putText(frame, f'Press "c" key to calibrate colors.', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(frame, f'Balls sequence: {res}', (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                
    
    if key == ord('q'):
        break
    
    if key == ord('c'):
        calibrated = False
        color_names = iter(colors.keys())
        color = next(color_names)
        
    cv2.imshow("Mask", merge_mask) 
    cv2.imshow("Camera", frame)

cam.release()
cv2.destroyAllWindows()
