import collections
import cv2
import os
import numpy as np

data_dir = 'CarLicensePlate/'
files = os.listdir(data_dir)

MIN_AREA, MAX_AREA = 80, 550
MIN_WIDTH, MIN_HEIGHT = 2, 7
MIN_RATIO, MAX_RATIO = 0.2, 2.3

MAX_DIAG_LENGTH = 4
MAX_ANGLE_DIFF = 9.0
MAX_AREA_DIFF = 0.5
MAX_WIDTH_DIFF = 0.9
MAX_HEIGHT_DIFF = 0.3
MIN_N_MATCHED = 3

def find_chars(contour_list):
    matched_result_idx = []

    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))

            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))

            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length * MAX_DIAG_LENGTH and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d in contour_list:
            if d['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

        recursive_contour_list = find_chars(unmatched_contour)

        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx

for file in files:
    image = cv2.imread(data_dir + file)

    height, width, channel = image.shape
    temp = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.ones((3, 18), np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 2, 2)
    gray = cv2.Laplacian(gray, cv2.CV_16S, 3)
    gray = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
    gray = cv2.Canny(gray, 128, 200, 3)
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, mask, 3)
    th_img = cv2.threshold(morph, 100, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_dict = []
    for cnt in contours:

        (x, y, w, h) = cv2.boundingRect(cnt)
        contours_dict.append({
            'cnt': cnt,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2),
            'pt1': (x, y),
            'pt2': (x + w, y + h)
        })

        range_contours = []

        for d in contours_dict:
            area = d['w'] * d['h']
            ratio = d['w'] / d['h']

            if d['h'] < d['w']:
                if (ratio > 1.5 and ratio < 7.0) and (area > 3500 and area < 16000):
                    #cv2.rectangle(image, d['pt1'], d['pt2'], (255, 0, 0), 2)
                    range_contours.append([d['pt1'], d['pt2']])

    gray1 = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(gray1, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray1, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(gray1, imgTopHat)
    gray1 = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    img_blurred = cv2.GaussianBlur(gray1, ksize=(5, 5), sigmaX=0)

    img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )

    contours1, _ = cv2.findContours(
        img_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    cv2.drawContours(temp_result, contours=contours1, contourIdx=-1, color=(255, 255, 255))

    contours_dict1 = []

    for cnt1 in contours1:

        (x1, y1, w1, h1) = cv2.boundingRect(cnt1)

        contours_dict1.append({
            'cnt': cnt1,
            'x': x1,
            'y': y1,
            'w': w1,
            'h': h1,
            'cx': x1 + (w1 / 2),
            'cy': y1 + (h1 / 2),
            'pt1': (x1, y1),
            'pt2': (x1 + w1, y1 + h1)
        })

        possible_contours = []

        cnt = 0

        #cv2.rectangle(image, pt1=(x1, y1), pt2=(x1 + w1, y1 + h1), color=(255, 255, 255), thickness=2)

        for cd in contours_dict1:
            area1 = cd['w'] * cd['h']
            ratio1 = cd['w'] / cd['h']

            if MIN_AREA < area1 < MAX_AREA and cd['w'] > MIN_WIDTH and cd['h'] > MIN_HEIGHT and MIN_RATIO < ratio1 < MAX_RATIO:
                cd['idx'] = cnt
                cnt += 1
                possible_contours.append(cd)

        for cd in possible_contours:
            cv2.rectangle(temp_result, cd['pt1'], cd['pt2'], (255, 255, 255), 2)

    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    index_contours = []

    for r in matched_result:
        for d in r:
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255), thickness=2)
            index_contours.append([(d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h'])])

    # result
    for i in range(len(range_contours)):
        cnt = 0
        max_counters = []
        for j in range(len(index_contours)):
            counters = {}
            if (index_contours[j][0][0] > range_contours[i][0][0]) and (index_contours[j][0][0] < range_contours[i][1][0]) and (index_contours[j][0][1] > range_contours[i][0][1]) and (index_contours[j][0][1] < range_contours[i][1][1]):
                #cv2.rectangle(image, pt1=range_contours[i][0], pt2=range_contours[i][1], color=(255, 0, 255), thickness=2)
                cnt += 1

            counters[i] = cnt
        print(counters[i])
        if max(list(counters.values())) > 4:
            b, g, r= cv2.split(image)
            cv2.rectangle(g, pt1=range_contours[i][0], pt2=range_contours[i][1], color=(255, 0, 0), thickness=-1)

            image = cv2.merge((r, g, b))

    cv2.imwrite('CarLicensePlate_result/' + file, image)
    cv2.imshow(file, image)
    cv2.waitKey(0)