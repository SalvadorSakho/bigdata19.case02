"""
Assignment 04
=============

The goal of this assignment is to ignore eye regions of interest (ROI) that are not placed within face ROI.

The code you will write in this file will be similar to main.py code, but will include additional rectangles filtering.

Run this code with

    > invoke run assignment04.py
"""
from main import MODEL_FACE, MODEL_EYE, time, cv2, tqdm


def main():
    camera = cv2.VideoCapture(0)
    while not camera.isOpened():
        time.sleep(0.2)

    try:
        with tqdm() as progress:
            while True:
                ret, frame = camera.read()
                cv2.imshow('Objects', process(frame))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                progress.update()
    finally:
        # gracefully close
        camera.release()
        cv2.destroyAllWindows()


def process(frame):
    """Process initial frame and tag recognized objects."""

    # 1. Convert initial frame to grayscale
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # For every model:
    color, parameters = ((255, 255, 0), {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)})

    # Находим лицо
    face = MODEL_FACE.detectMultiScale(grayframe, **parameters)

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        eyes_in_frame = grayframe[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = MODEL_EYE.detectMultiScale(eyes_in_frame)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return frame


if __name__ == '__main__':
    main()
