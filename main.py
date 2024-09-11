import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

frameWidth = 1920
frameHeight = 1080
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB) 
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
    
    cTime = time.time() 
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    # Show the current thickness and radius of the circles
    cv2.putText(img, f'Thickness: {drawSpec.thickness}', (20, 110), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(img, f'Circle Radius: {drawSpec.circle_radius}', (20, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('w'):
        drawSpec.thickness += 1
    elif key == ord('s') and drawSpec.thickness > 1:
        drawSpec.thickness -= 1
    elif key == ord('a'):
        drawSpec.circle_radius += 1
    elif key == ord('d') and drawSpec.circle_radius > 1:
        drawSpec.circle_radius -= 1
cap.release()