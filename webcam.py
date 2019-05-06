import cv2

video_capture = cv2.VideoCapture(0)

while video_capture.isOpened():
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # # Draw a rectangle
    # for (x, y, w, h) in faces:
    x = 50
    y = 100
    w = 200
    h = 200
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
    # cv2.rectangle(frame,(900,100),(1300,500),(0,0,0),3)

    hand = frame[y:y + h, x:x + w]
    # adaptive_thresh_gaussian = cv2.adaptiveThreshold(grey_hand, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Otsu's thresholding after Gaussian filtering
    grey_hand = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey_hand, (5, 5), 0)
    ret_otsu_threshold, otsu_threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # hand_ycrcb = cv2.cvtColor(hand, cv2.COLOR_BGR2YCR_CB) # https://medium.com/@danojadias/what-is-ycbcr-964fde85eeb3
    # blur = cv2.GaussianBlur(hand_ycrcb, (11, 11), 0)
    # adaptive_thresh_gaussian = cv2.threshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    # adaptive_thresh_gaussian = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Display the resulting frame
    cv2.imshow('Video', otsu_threshold)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
