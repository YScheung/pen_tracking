import cv2
import numpy as np
import time

canvasH, canvasW = 1000, 1414
coordinates = []
track_img = None

def get_paper():
    cam = cv2.VideoCapture(0)
    
    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame,1)

        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "white_paper.png"
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()


def get_coordinates():
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:

            print(x, ' ', y)
            coordinates.append([x,y])
    
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(x) + ',' +
                        str(y), (x,y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image', img)
    
        if event==cv2.EVENT_RBUTTONDOWN:
    
            print(x, ' ', y)
            coordinates.append([x,y])
    
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = img[y, x, 0]
            g = img[y, x, 1]
            r = img[y, x, 2]
            cv2.putText(img, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x,y), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image', img)

    img = cv2.imread('white_paper.png', 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_paper_corners(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    frame_r, frame_c, _ = frame_hsv.shape

    # tunable params 
    sensitivity = 30
    GREEN_MIN = np.array([60 - sensitivity, 50, 50], np.uint8)
    GREEN_MAX = np.array([60 + sensitivity, 255, 255], np.uint8)
    BLUE_MIN = np.array([75, 100, 0], np.uint8)
    BLUE_MAX = np.array([125, 200, 200], np.uint8)

    frame_threshed = cv2.inRange(frame_hsv, BLUE_MIN, BLUE_MAX)

    top_left_row, top_left_col = np.nonzero(frame_threshed[0:(frame_r//2), 0:(frame_c//2)])
    top_right_row, top_right_col = np.nonzero(frame_threshed[0:(frame_r//2), (frame_c//2):frame_c])
    bottom_left_row, bottom_left_col = np.nonzero(frame_threshed[(frame_r//2):frame_r, 0:(frame_c//2)])
    bottom_right_row, bottom_right_col = np.nonzero(frame_threshed[(frame_r//2):frame_r, (frame_c//2):frame_c])

    top_left = (np.mean(top_left_col), np.mean(top_left_row))
    top_right = (np.mean(top_right_col) + frame_c//2, np.mean(top_right_row))
    bottom_left = (np.mean(bottom_left_col), np.mean(bottom_left_row) + frame_r//2)
    bottom_right = (np.mean(bottom_right_col) + frame_c//2, np.mean(bottom_right_row) + frame_r//2)

    paper_corners = [bottom_left, bottom_right, top_left, top_right]

    return frame_threshed, paper_corners

def get_projection_matrix(paper_corners):
    """
    paper_corners points of the paper: 
        top-left, top-right, bottom-left, bottom-right
    The pen on camera:
        bottom-left, bottom-right, top-left, top-right
    """
    
    canvas_corners = np.array([[0,0], [canvasW-1,0], [0, canvasH-1], [canvasW-1, canvasH-1]])

    # A is the linear equations to be solved
    A = np.empty((8,9))
    for pt in range(4):
        x_l, y_l = paper_corners[pt]
        x_r, y_r = canvas_corners[pt]
        # each point corresponds to 2 rows of the matrix A
        A[pt*2] = [  0,   0, 0, x_l, y_l, 1, -y_r * x_l, -y_r * y_l, -y_r]
        A[pt*2 + 1] = [x_l, y_l, 1,   0,   0, 0, -x_r * x_l, -x_r * y_l, -x_r]
    U, s, V = np.linalg.svd(A)
    H = V[-1].reshape(3,3) # homography matrix

    return H

def paper_to_canvas(H, paper_point):
    x, y = paper_point
    canvas_pt = H @ np.array([x,y,1]).T
    scale = 1 / canvas_pt[2]
    canvas_pt *= scale

    return int(canvas_pt[0]), int(canvas_pt[1])

def calibration():
    video = cv2.VideoCapture(0)

    while video.isOpened():
        isTrue, frame = video.read()
        if not isTrue: break
        frame = cv2.flip(frame,1)
        frame = frame[min(coordinates[0][1],coordinates[1][1])-50:max(coordinates[2][1],coordinates[3][1])+50,min(coordinates[0][0],coordinates[2][0])-50:max(coordinates[1][0],coordinates[3][0])+50]
        frame_threshed, paper_corners = find_paper_corners(frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255,0,0)
        thickness = 2
        for pt in paper_corners:
            try:
                pt = int(pt[0]), int(pt[1])
                cv2.putText(frame, str(pt), pt ,font, font_scale, color, thickness, cv2.LINE_AA)
            except:
                pass

        cv2.imshow("paper_corner", frame_threshed)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(20) & 0xFF==ord('q'):
            break 
    H = get_projection_matrix(paper_corners)
    video.release()
    cv2.destroyAllWindows()
    return H


def get_bbox():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        frame = frame[min(coordinates[0][1],coordinates[1][1])-50:max(coordinates[2][1],coordinates[3][1])+50,min(coordinates[0][0],coordinates[2][0])-50:max(coordinates[1][0],coordinates[3][0])+50]
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "pen_setup.png"
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()

    #cap = cv2.VideoCapture(0)
    tracker = cv2.legacy_TrackerMOSSE.create()
    #tracker = cv2.TrackerGOTURN.create()
    #tracker = cv2.legacy_TrackerCSRT.create()
    #tracker = cv2.legacy_TrackerKCF.create()
    #success, img = cap.read()
    img = cv2.imread("pen_setup.png")
    bbox = cv2.selectROI("Tracking", img, False)
    tracker.init(img, bbox)


    return tracker


lines = []
def track_pen_motion(tracker):
    cap = cv2.VideoCapture(0)
    canvas = None
    x1,y1=0,0
    track = False

    prev_frame_time = 0
    new_frame_time = 0

    while(1):
        timer = cv2.getTickCount()
        success, img = cap.read()
        img = cv2.flip(img,1)
        img = img[min(coordinates[0][1],coordinates[1][1])-50:max(coordinates[2][1],coordinates[3][1])+50,min(coordinates[0][0],coordinates[2][0])-50:max(coordinates[1][0],coordinates[3][0])+50]

        success,bbox = tracker.update(img)
        x,y,w,h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if success:
            drawBox(img,bbox)
        else:
            cv2.putText(img, "Lost", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))

        cv2.putText(img, str(int(fps)), (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)


        _, frame = cap.read()
        frame = cv2.flip( frame, 1 )
        print(frame.shape)

        key_input = cv2.waitKey(1) & 0xFF
        if key_input == ord('b'):
            track = True
            
        if canvas is None:
            canvas = np.zeros((frame.shape[0], frame.shape[1]))

        if track:
            cv2.putText(img, "Start signing", (75,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            x2,y2,w,h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            x2 = int(x2 + w/2)
            y2 = int(y2 + h/2)

            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2     
            else:
                canvas = cv2.line(canvas, (x1,y1),(x2,y2), [0,0,255], 2)

            lines.append([x1,y1,x2,y2])
            x1,y1= x2,y2

        cv2.imshow("Tracking",img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord('c'):
            canvas = None
            track = False
            lines.clear()

    cap.release()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def translate(H):
    canvas = np.uint8(np.ones((canvasH,canvasW,3)) * 255)
    canvas_original = np.uint8(np.ones((canvasH,canvasW,3)) * 255)
    print("LINES", lines)

    for i in range(len(lines)):
        penX,penY = lines[i][0],lines[i][1]
        penX1, penY1 = lines[i][2], lines[i][3]
        transX, transY = paper_to_canvas(H, (penX,penY))
        transX1, transY1 = paper_to_canvas(H, (penX1,penY1))
        canvas = cv2.line(canvas, (transX, transY), (transX1, transY1), [255,0,0], 3)
        canvas_original = cv2.line(canvas_original, (penX, penY), (penX1, penY1), [255,0,0], 3)

    cv2.imwrite("canvas_frame.png", canvas)
    cv2.imwrite("canvas_frame_og.png", canvas_original)



def drawBox(img, bbox):
    x,y,w,h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x,y),((x+w), (y+h)),(255,0,255),3,1)
    cv2.putText(img, "Tracking", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)



if __name__ == "__main__":
    get_paper()
    get_coordinates()
    H = calibration()
    tracker = get_bbox()
    track_pen_motion(tracker)
    translate(H)
