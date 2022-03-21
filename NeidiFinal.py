from cv2 import cv2
import imutils as imutils
import numpy as np
from Shape import ShapeFunction 



path_to_video = "wooden_blockOriginal.mp4"
ShapeFunction(path_to_video)

cap = cv2.VideoCapture(path_to_video)

x1 = []
y2 = []
w1 = []
h1 = []

sequence = True
sequenceArray = []

# for i in range(0, 400):
#     _, img1 = cap.read()

x1Red = []
y2Red = []
w1Red = []
h1Red = []

x1Yellow = []
y2Yellow = []
w1Yellow = []
h1Yellow = []

_, img = cap.read()
# img1 = cap.read()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 132, 0])
upper_red = np.array([16, 255, 255])
mask_red = cv2.inRange(hsv, lower_red, upper_red)
cnts_red = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts_red = imutils.grab_contours(cnts_red)

lower_yellow = np.array([9, 165,  57])
upper_yellow = np.array([28, 255, 255])
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
cnts_yellow = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts_yellow = imutils.grab_contours(cnts_yellow)

lower_blue = np.array([95, 100,  -4])
upper_blue = np.array([119, 255, 80])
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
cnts_blue = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts_blue = imutils.grab_contours(cnts_blue)



for c in cnts_red:
    area1 = cv2.contourArea(c)
    approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)

    if area1 > 500:
        cv2.drawContours(img, [c], -1, (0, 255, 0), 3)

        M = cv2.moments(c)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.circle(img, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(img, "Red", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        x1Red.append(cv2.boundingRect(approx)[0])
        y2Red.append(cv2.boundingRect(approx)[1])
        w1Red.append(cv2.boundingRect(approx)[2])
        h1Red.append(cv2.boundingRect(approx)[3])


for c in cnts_yellow:
    area2 = cv2.contourArea(c)
    approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
    if area2 > 10:
        cv2.drawContours(img, [c], -1, (0, 255, 0), 3)

        M = cv2.moments(c)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.circle(img, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(img, "Yellow", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        x1Yellow.append(cv2.boundingRect(approx)[0])
        y2Yellow.append(cv2.boundingRect(approx)[1])
        w1Yellow.append(cv2.boundingRect(approx)[2]+20)
        h1Yellow.append(cv2.boundingRect(approx)[3]+20)

for c in cnts_blue:
    area3 = cv2.contourArea(c)
    approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
    if area3 > 40:
        cv2.drawContours(img, [c], -1, (0, 255, 0), 3)

        M = cv2.moments(c)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.circle(img, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(img, "Blue", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        x1.append(cv2.boundingRect(approx)[0])
        y2.append(cv2.boundingRect(approx)[1])
        w1.append(cv2.boundingRect(approx)[2])
        h1.append(cv2.boundingRect(approx)[3])




countBlue = len(x1)
countRed = len(x1Red)
countYellow = len(x1Yellow)



trackersForRed = []
trackersForYellow = []

UpdatedtrackerYellow = []
Yellow_Red_Trackers = []



for i in range(0, countRed):
    trackersForRed.append(cv2.TrackerCSRT_create())

for i in range(0, countYellow):   
    trackersForYellow.append(cv2.TrackerCSRT_create())


bbox = []
bboxRed = []
bboxYellow = []

tempYellow = []

for i in range(0, countBlue):
    bbox.append([x1[i], y2[i], w1[i], h1[i]])

for i in range(0, countRed):
    if y2Red[i]>=150:
        bboxRed.append([x1Red[i], y2Red[i], w1Red[i], h1Red[i]])
        # print("BBOX: ", bboxRed)
        # exit()
        trackersForRed[i].init(img, bboxRed[i])



for i in range(0, countYellow):
    
    if x1Yellow[i]>406 and x1Yellow[i] != 872 :
        tempYellow.append([x1Yellow[i], y2Yellow[i], w1Yellow[i], h1Yellow[i]])

    
UpdatedtrackerYellow_COUNT = len(UpdatedtrackerYellow)
tempcountYellow = len(tempYellow)
bboxYellow = tempYellow.copy()


for i in range(0, tempcountYellow):

    trackersForYellow[i].init(img, bboxYellow[i])
        
Yellow_Red_Trackers = bboxYellow.copy()
Yellow_Red_Trackers.extend(bboxRed)
Yellow_Red_Trackers.sort(reverse=True)

countCombination = len(Yellow_Red_Trackers)
bluearray = bbox.sort(reverse=True)


def drawBox(img, bbox, bboxRed, bboxYellow):

    global sequence
    global sequenceArray


    # print("BBOXBlue: ", bbox)
    # print("BBOXRed: ", bboxRed)
    # print("BBOXYellow: ", bboxYellow)
    # exit()

    x = []
    y = []
    w = []
    h = [] 

    xRed = []
    yRed = []
    wRed = []
    hRed = []

    xYellow = []
    yYellow = []
    wYellow = []
    hYellow = []

    topXValue = [0,0,0,0,0]
    bottomXValue = [0,0,0,0,0]

    def checkLength():
        if len(bottomXValue) != len(topXValue):
            if len(bottomXValue) < len(topXValue):
                while len(bottomXValue) < len(topXValue):
                    bottomXValue.append(0)
            elif len(topXValue) < len(bottomXValue):
                while len(topXValue) < len(bottomXValue):
                    topXValue.append(0)

    
    print("TOPLENGTH: ", len(topXValue), "BottomLEngth: ", len(bottomXValue))

    for i in range(0, len(bbox)):
        x.append(bbox[i][0])
        y.append(bbox[i][1])
        w.append(bbox[i][2])
        h.append(bbox[i][3])

    

    for i in range(0, len(bboxRed)):
        xRed.append(bboxRed[i][1][0])
        yRed.append(bboxRed[i][1][1])
        wRed.append(bboxRed[i][1][2])
        hRed.append(bboxRed[i][1][3])

    
        
    for i in range(0, len(bboxYellow)):
        xYellow.append(bboxYellow[i][1][0])
        yYellow.append(bboxYellow[i][1][1])
        wYellow.append(bboxYellow[i][1][2])
        hYellow.append(bboxYellow[i][1][3])

    tempArray1 = [0] * (len(bboxYellow))
    

    for i in range(0, len(bboxYellow)):
        cv2.rectangle(img, (xYellow[i], yYellow[i]), ((xYellow[i] + wYellow[i]), (yYellow[i] + hYellow[i])), (255, 0, 0), 3, 1)
        cv2.putText(img, "Yellow"+str(i)+"-"+str(xYellow[i])+","+str(yYellow[i]), (xYellow[i], int(yYellow[i]+(hYellow[i]/2))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
        tempArray1[len(bboxYellow)-i-1] = xYellow[i]
    
    tempArray2 = [0] * (len(bboxRed))

    for i in range(0, len(bboxRed)):
        cv2.rectangle(img, (xRed[i], yRed[i]), ((xRed[i] + wRed[i]), (yRed[i] + hRed[i])), (255, 0, 0), 3, 1)
        cv2.putText(img, "Red"+str(i)+"-"+str(xRed[i])+","+str(yRed[i]), (xRed[i], int(yRed[i]+(hRed[i]/2))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        tempArray2[len(bboxRed)-i-1] = xRed[i]
    
    tempArray3 = tempArray1 + tempArray2
    topXValue = tempArray3

    for i in range(0, len(bbox)):
        cv2.rectangle(img, (x[i], y[i]), ((x[i] + w[i]), (y[i] + h[i])), (255, 0, 0), 3, 1)
        cv2.putText(img, "Blue"+str(i)+"-"+str(x[i])+","+str(y[i]), (x[i], int(y[i]+(h[i])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        checkLength()
        try:
            bottomXValue[i] = x[i]
        except:
            print("Could not detect the shapes properly")
    
    

    print(sequenceArray)
    if len(sequenceArray) < len(topXValue):
        for i in range(0, len(topXValue)):
            checkLength()
            print("TOPLENGTH: ", len(topXValue), "BottomLEngth: ", len(bottomXValue))
            diff = abs(topXValue[i]-bottomXValue[i])
            if diff < 50:
                if i not in sequenceArray:
                    sequenceArray.append(i)
    else:
        sequence = all(sequenceArray[i] <= sequenceArray[i+1] for i in range(len(sequenceArray)-1))
        if sequence == True:
            cv2.putText(img, "Sequence Correct", (75, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(img, "Sequence False", (75, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    


counter2 = 0
while True:
    # if(counter2 == 380):
    #     for i in range(0, 700):

    #         _, img1 = cap.read()


    timer = cv2.getTickCount()
    success, img = cap.read()

    
    
    for i in range(0, len(bboxRed)):
        bboxRed[i] = trackersForRed[i].update(img)

    for i in range(0, len(bboxYellow)):
        bboxYellow[i] = trackersForYellow[i].update(img)


    drawBox(img, bbox,  bboxRed, bboxYellow)



    fps= cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(img, str(fps),(75,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Tracking",img)

    if cv2.waitKey(1) == 27:
        break
    counter2 = counter2 + 1