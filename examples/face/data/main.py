import cv2
import numpy as np

capture = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('/Users/eidelen/dev/libs/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('/Users/eidelen/dev/libs/opencv-3.2.0/data/haarcascades/haarcascade_eye.xml')

f = open('adrian.csv', 'w')

addFaceToFile = False

while(True):

    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:

        [x,y,w,h] = faces[0]
        roi_gray = gray[y:y + h, x:x + w]

        # in face, find both eyes
        eyes = eyes_cascade.detectMultiScale(roi_gray)

        if len(eyes) > 1:
            [e1x, e1y, e1w, e1h] = eyes[0]
            [e2x, e2y, e2w, e2h] = eyes[1]

            eye_dist = abs((e1x+(e1w/2.0) - (e2x+(e2w/2.0))))
            cut_cent_x = (e1x+(e1w/2.0) + e2x+(e2w/2.0)) / 2.0
            cut_cent_y = (e1y+(e1h/2.0) + e2y+(e2h/2.0)) / 2.0

            region_x = int(round(cut_cent_x - 0.8 * eye_dist))
            region_w = int(round(1.6 * eye_dist))

            region_y = int(round(cut_cent_y - eye_dist*0.2))
            region_h = int(region_w)

            reg_interest = roi_gray[region_y:region_y + region_h, region_x:region_x + region_w]

            if addFaceToFile:
                cv2.rectangle(frame, (x+region_x, y+region_y), (x+region_x+region_w, y+region_y+region_h), (0, 255, 0), 4)
            else:
                cv2.rectangle(frame, (x + region_x, y + region_y), (x + region_x + region_w, y + region_y + region_h),
                              (255, 0, 0), 4)

            # draw both eyes
            cv2.rectangle(frame, (x+e1x, y+e1y), (x+e1x + e1w, y+e1y + e1h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x + e2x, y + e2y), (x + e2x + e2w, y + e2y + e2h), (0, 0, 255), 2)

            [reg_height, reg_width] = reg_interest.shape

            if reg_height > 0:
                if reg_width > 0:
                    cv2.imshow('face', reg_interest)

            # safe reg_interest to file
            if addFaceToFile:
                scaledFace = cv2.resize(reg_interest, (64, 64), interpolation=cv2.INTER_CUBIC)
                cv2.imshow('face_scaled', scaledFace)

                normVals = (scaledFace / 128.0) - 1.0

                x_str = ''
                for pVal in np.nditer(normVals):
                    x_str = x_str + str(pVal) + ', '

                f.write(x_str);
                f.write('\n')
                addFaceToFile = False



    cv2.imshow('frame', frame)

    keyCode = cv2.waitKey(1)
    if keyCode == ord('q'):
        break;
    if keyCode == ord('a'):
        addFaceToFile = True


capture.release()
cv2.destroyAllWindows()

f.close();