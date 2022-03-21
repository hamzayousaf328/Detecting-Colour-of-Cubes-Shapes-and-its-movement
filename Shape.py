from cv2 import cv2

import numpy as np

def ShapeFunction(path):

    cap = cv2.VideoCapture(path)
    _, image = cap.read()
    # path_to_image= image
    # image = cv2.imread()
    
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find Canny edges
    edged = cv2.Canny(gray, 30, 200)
    cv2.waitKey(0)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #For Contours
    # boundRect = [None]*len(contours)
    # for i, c in enumerate(contours):
    #     contours_poly = cv2.approxPolyDP(c, 3, True)
    #     boundRect[i] = cv2.boundingRect(contours_poly)

    # for i in range(len(contours)):
    #     cv2.rectangle(image, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0,0,0), 2)
    #Finish For Contours
    

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        print("ApPROX: %d" %len(approx))
        cv2.drawContours(image, [approx], 0, (0, 0, 0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if len(approx) > 10:
            print("Sphere")
            cv2.drawContours(image, [contour], -1, (255, 0, 0), 3)
            cv2.putText(image, 'Cube', (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, [0, 0, 0], 2)

        elif len(approx) == 6 or len(approx) == 7:
            print("Cube")
            cv2.drawContours(image, [contour], -1, (255, 0, 0), 3)
            x1, y2, w1, h1 = cv2.boundingRect(approx)
            print("Dimensions: %d, %d, %d, %d " % (x1, y2, w1, h1))
            cv2.putText(image, 'Cube ', (x, y), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, [0, 0, 0], 2)


        elif len(approx) == 10 or len(approx) == 9:
            print("Cylinder")
            cv2.drawContours(image, [contour], -1, (255, 0, 0), 3)
            cv2.putText(image, 'Cylinder', (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, [0, 0, 0], 2)



    cv2.imshow('Canny Edges After Contouring', edged)
    cv2.waitKey(0)

  
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    cv2.imshow('Contours', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return contours