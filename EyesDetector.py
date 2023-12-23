import numpy as np, os, cv2

import cv2
def getAccuracy(detected, gt):
    test = cv2.bitwise_and(detected, gt)
    total = cv2.countNonZero(gt)
    right = cv2.countNonZero(test)
    return right / total

def getAvgAcc(resultPath, gtPath):
    sum = 0
    for i in range(1, 21):
        sum += getAccuracy(cv2.imread(resultPath + str(i) + ".bmp", cv2.IMREAD_GRAYSCALE), cv2.imread(gtPath + str(i) + ".tif", cv2.IMREAD_GRAYSCALE))
    return sum / 20.0

data_dir = 'Eyes/images/'
#true_dir = 'Eyes/result/'
files = os.listdir(data_dir)
#trues = os.listdir(true_dir)
print(files)

for n, file in enumerate(files):
    image = cv2.imread(data_dir + file)
    #n += 1
    #cv2.imshow('true', cv2.imread(true_dir + trues[n]))
    cv2.imshow('image', image)

    #image = cv2.blur(image, (5, 5))
    #image = cv2.medianBlur(image, 5)
    image = cv2.GaussianBlur(image, (5, 5), 3)

    b, g, r = cv2.split(image)

    clahe = cv2.createCLAHE(clipLimit=2.3, tileGridSize=(5, 5))
    clahe_g = clahe.apply(g)

    r1 = cv2.morphologyEx(clahe_g, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    f4 = cv2.subtract(R3, clahe_g)
    f5 = clahe.apply(f4)

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (5, 5))
    #morph = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)

    #division = cv2.divide(gray, morph, scale = 255)
    #thresh = cv2.threshold(division, 0, 255, cv2.THRESH_OTSU)[1]
    #thresh = 255 - thresh

    #mask = np.zeros_like(thresh)
    #contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours = contours[0] if len(contours) == 2 else contours[1]

    #area_thresh = 10000
    #for cntr in contours:
    #    area = cv2.contourArea(cntr)
    #    if area > area_thresh:
    #        cv2.drawContours(mask, [cntr], -1, 255, 2)

    #result1 = cv2.bitwise_and(thresh, mask)
    #mask = cv2.merge([mask, mask, mask])
    #result2 = cv2.bitwise_and(image, mask)

    #thresh = cv2.erode(thresh, kernel, iterations=1)
    #thresh = cv2.dilate(thresh, kernel, iterations=1)

    def erode(image):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 2))
        erosion = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        erosion = cv2.erode(erosion, kernel, iterations=1)
        return erosion

    #gray = erode(image)

    image = cv2.blur(image, (5, 5))
    #image = cv2.medianBlur(image, 5)
    #image = cv2.GaussianBlur(image, (5, 5), 3)

    gray = cv2.threshold(f5, 50, 255, cv2.THRESH_BINARY)[1]
    mask = np.ones((5, 11), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, mask, iterations = 1)

    cv2.imshow('final', gray)
    cv2.imwrite('Eyes/processing/' + file, gray)
    cv2.waitKey(0)

# 정확도 평가
accuracy = getAvgAcc('Eyes/result/', 'Eyes/processing/')
print(accuracy)