import numpy as np
import cv2 as cv

def initialize_video_processing(video_path, frame_width, frame_height):
    cap = cv.VideoCapture(video_path)
    size = (frame_width, frame_height)
    fgbg = cv.createBackgroundSubtractorMOG2()
    feature_params = dict(maxCorners=1, qualityLevel=.6, minDistance=25, blockSize=9)
    result = cv.VideoWriter('op.mp4', cv.VideoWriter_fourcc(*'MJPG'), 10, size)
    return cap, fgbg, feature_params, result, size

def process_frame(cap, fgbg, feature_params, result, frame_width, frame_height):
    path_points = []
    left_threshold = frame_width * 0.1
    right_threshold = frame_width * 0.9

    while True:
        ret, oframe = cap.read()
        if oframe is None:
            break
        oframe = cv.resize(oframe, (frame_width, frame_height))
        mask = fgbg.apply(oframe)
        frame = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))

        ball = cv.goodFeaturesToTrack(frame, **feature_params)
        if ball is not None:
            x, y = ball[0][0]
            current_position = (int(x), int(y))
            path_points.append(current_position)
            cv.circle(oframe, current_position, 8, (180, 180, 0), 2)

            if current_position[0] >= right_threshold or current_position[0] <= left_threshold:
                path_points = []  # Reset path points when the ball reaches a side

        for i in range(1, len(path_points)):
            cv.line(oframe, path_points[i-1], path_points[i], (0, 255, 0), 2)

        cv.imshow("Track", oframe)
        result.write(oframe)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break

def cleanup(cap, result):
    result.release()
    cap.release()
    cv.destroyAllWindows()

def main():
    frame_width = 640
    frame_height = 480
    # video_path =
    cap, fgbg, feature_params, result, size = initialize_video_processing(video_path, frame_width, frame_height)
    process_frame(cap, fgbg, feature_params, result, frame_width, frame_height)
    cleanup(cap, result)

if __name__ == "__main__":
    main()
