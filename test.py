import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from tracker import*

model=YOLO('best.pt')

# Mouse event function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
            
        


#provide input video
cap=cv2.VideoCapture('cr.mp4')


my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)


tracker=Tracker()

count=0
area1=[(544,12),(587,377),(713,372),(643,13)]
area2=[(763,17),(969,343),(1016,298),(924,16)]

area1item=[(493,210),(558,208),(573,365),(497,368)]
area2item=[(838,194),(867,191),(956,328),(889,342)]

while True:    
    ret,frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
    # print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
   

#    print(px)
    list1=[]
    list2=[]
    itemlist1=[]
    itemlist2=[]
    
    for index,row in px.iterrows():
        # print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        print(d)
        c=class_list[d]
        if 'person' in c:
            cx=int(x1+x2)//2            # center point of object
            cy=int(y1+y2)//2
            w,h=x2-x1,y2-y1
            result=cv2.pointPolygonTest(np.array(area1, np.int32), ((cx,cy)), False)
            if result>=0:
        #        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),-1)
                cvzone.cornerRect(frame,(x1,y1,w,h),3,2)
                cv2.circle(frame,(cx,cy),4,(255,255,0),-1)
                cvzone.putTextRect(frame,f'person',(x1,y1),1,1,(0,0,0),(255,255,0))
                list1.append(cx)
            result1=cv2.pointPolygonTest(np.array(area2, np.int32), ((cx,cy)), False)
            if result1>=0:
        #        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),-1)
                cvzone.cornerRect(frame,(x1,y1,w,h),3,2)
                cv2.circle(frame,(cx,cy),4,(255,255,0),-1)
                cvzone.putTextRect(frame,f'person',(x2,y2),1,1,(0,0,0),(255,255,0))
                list2.append(cx)
        
        if 'item' in c:
            
            cx=int(x1+x2)//2            # center point of object
            cy=int(y1+y2)//2
            w,h=x2-x1,y2-y1
            result=cv2.pointPolygonTest(np.array(area1item, np.int32), ((cx,cy)), False)
            if result>=0:
                #        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),-1)
                # cvzone.cornerRect(frame,(x1,y1,w,h),3,2)
                # cv2.circle(frame,(cx,cy),4,(0,255,0),-1)
                # cvzone.putTextRect(frame,f'item',(x1,y1),1,1,3,(255,255,255))
                itemlist1.append([x1,y1,x2,y2])

            result1=cv2.pointPolygonTest(np.array(area2item, np.int32), ((cx,cy)), False)
            if result1>=0:
        #        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),-1)
                # cvzone.cornerRect(frame,(x1,y1,w,h),3,2)
                # cv2.circle(frame,(cx,cy),4,(0,255,0),-1)
                # cvzone.putTextRect(frame,f'item',(x2,y2),1,1,3,(255,255,255))
                itemlist2.append([x1,y1,x2,y2])

        # main_bbox_idx= tracker.update(list1)     # personlist in first couter id       
        # for bbox in main_bbox_idx:
            item1count=[]
            item2count=[]
            sub_bbox_idx1= tracker.update(itemlist1)  
            for bbox in sub_bbox_idx1:
                    x3, y3, x4, y4,id1=bbox 
                    cx3=int(x3+x4)//2
                    cy3=int(y3+y4)//2                   
                    cv2.circle(frame, (cx3, cy3), 4, (255,0,0),-1)
                    cv2.rectangle(frame, (x3, y3), (x4,y4), (0,0,0),1)
                    cvzone.putTextRect (frame,f'item',(x3,y3),1,1,(0,0,0),(0,255,255))
                    if item1count.count(id1)==0:
                        item1count.append(id1)
            
            sub_bbox_idx2= tracker.update(itemlist2)  
            for bbox in sub_bbox_idx2:
                    x3, y3, x4, y4,id1=bbox 
                    cx3=int(x3+x4)//2
                    cy3=int(y3+y4)//2                   
                    cv2.circle(frame, (cx3, cy3), 4, (255,0,0),-1)
                    cv2.rectangle(frame, (x3, y3), (x4,y4), (0,0,0),1)
                    cvzone.putTextRect (frame,f'item',(x3,y3),1,1,(0,0,0),(0,255,255))
                    if item2count.count(id1)==0:
                        item2count.append(id1)


    cr1=len(list1)    
    cr2=len(list2)
    item1counter=len(item1count)
    item2counter=len(item2count)

    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,255,255),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,255,255),2)
    
    cvzone.putTextRect(frame, f' counter1: {cr1}', (20,30),1,1,(0,0,0),(255,255,255))
    cvzone.putTextRect(frame, f' counter2: {cr2}', (20,50),1,1,(0,0,0),(255,255,255))
    cvzone.putTextRect(frame, f' counter1 items: {item1counter}', (20,70),1,1,(0,0,0),(255,255,255))
    cvzone.putTextRect(frame, f' counter2 items: {item2counter}', (20,90),1,1,(0,0,0),(255,255,255))
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()


