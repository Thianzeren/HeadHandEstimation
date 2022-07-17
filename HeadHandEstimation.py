import cv2
import tkinter as tk
import time
import atexit
import mediapipe as mp
import numpy as np
from threading import Thread
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

from ActionProfiles import ActionProfiles as ap

def exit_handler(): # On exit release all keys
    print('Application is ending, released all keys')

    for key, action in keyPressDict.items():
        if(action != ""):
            # Release previous keys
            keyboard.release(action)

def initCamDetection():

    print("Initializing camera, cv2 and mediapipe set up...")

    def handleKeyChanges(kpDict):
        global keyPressDict

        for key, currAction in kpDict.items():
            prevAction = keyPressDict[key]
            if(prevAction != currAction):

                if(prevAction != ""):
                    # Release previous keys
                    keyboard.release(prevAction)

                if(currAction != ""):
                    # Press new key
                    keyboard.press(currAction)

                # Set new key
                keyPressDict[key] = currAction

    def runHeadPoseEstimation():
        global movementText
        global headPoseCoords
        global actionDict

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if faceResults.multi_face_landmarks:
            for face_landmarks in faceResults.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x*img_w, lm.y*img_h)
                            nose_3d = (lm.x*img_w, lm.y*img_h, lm.z*8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array (face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1*img_w

                cam_matrix = np.array([[focal_length, 0, img_h /2],[0,focal_length, img_w /2],[0,0,1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)
                
                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                y = angles[0] * 360
                x = angles[1] * 360

                # Head Tilt Detection Coordinates
                topLeftX = headPoseCoords["topLeftX"]
                topLeftY = headPoseCoords["topLeftY"]
                btmLeftX = headPoseCoords["btmLeftX"]
                btmLeftY = headPoseCoords["btmLeftY"]
                topRightX = headPoseCoords["topRightX"]
                topRightY = headPoseCoords["topRightY"]
                btmRightX = headPoseCoords["btmRightX"]
                btmRightY = headPoseCoords["btmRightY"]
                leftX = headPoseCoords["leftX"]
                rightX = headPoseCoords["rightX"]
                downY = headPoseCoords["downY"]
                upY = headPoseCoords["upY"]

                keyChanges = {}
                
                # See where the user's head tilting
                if x < topLeftX and y > topLeftY:
                    keyChanges.update([("kpX",actionDict["leftMove"]),("kpY",actionDict["upMove"])])
                    movementText = "TopLeft"
                elif x < btmLeftX and y < btmLeftY:
                    keyChanges.update([("kpX",actionDict["leftMove"]),("kpY",actionDict["downMove"])])
                    movementText = "BtmLeft"
                elif x > topRightX and y > topRightY:
                    keyChanges.update([("kpX",actionDict["rightMove"]),("kpY",actionDict["upMove"])])
                    movementText = "TopRight"
                elif x > btmRightX and y < btmRightY:
                    keyChanges.update([("kpX",actionDict["rightMove"]),("kpY",actionDict["downMove"])])
                    movementText = "BtmRight"
                elif x < leftX:
                    keyChanges.update([("kpX", actionDict["leftMove"]), ("kpY","")])
                    movementText = "Left"
                elif x > rightX:
                    keyChanges.update([("kpX", actionDict["rightMove"]), ("kpY","")])
                    movementText = "Right"
                elif y < downY:
                    keyChanges.update([("kpX", ""), ("kpY",actionDict["downMove"])])
                    movementText = "Down"
                elif y > upY:
                    keyChanges.update([("kpX", ""), ("kpY",actionDict["upMove"])])
                    movementText = "Up"
                else:
                    keyChanges.update([("kpX",""),("kpY","")])
                    movementText = "Forward"
                
                handleKeyChanges(keyChanges)
                
                angleText = "X:" + str(x)[0:3] + "Y:" + str(y)[0:3]

                addHeadDetectionImage(image, movementText, angleText, keyChanges["kpX"], keyChanges["kpY"], nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix, nose_2d)

    def runHandGestureEstimation():
        global actionText

        if handsResults.multi_hand_landmarks:
            
            # Draw landmarks
            for hand_landmarks in handsResults.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # Conditions
                # y coordinate: top is 0, btm is 1
                # x coordinate: left is 0, right is 1
                thumbCond = thumb_tip.x < thumb_mcp.x
                indexCond = index_tip.y > index_dip.y
                middleCond = middle_tip.y > middle_mcp.y
                ringCond = ring_tip.y > ring_mcp.y
                pinkyCond = pinky_tip.y > pinky_mcp.y
                
                # FOR LEFT HAND
                keyChanges = {}

                if(thumbCond and indexCond and middleCond and ringCond and pinkyCond):
                    keyChanges.update([("kpA1", "") , ("kpA2", ""), ("kpA3",""), ("kpA4", ""), ("kpA5", ""), ("kpA6", actionDict["fistAction"])])
                    actionText = "fist"
                else:
                    keyChanges.update({"kpA6" : ""})

                    if(thumbCond):
                        keyChanges.update({"kpA1" : actionDict["thumbAction"]})
                        actionText = "thumb"
                    else:
                        keyChanges.update({"kpA1" : ""})

                    if(indexCond):
                        keyChanges.update({"kpA2" : actionDict["indexAction"]})
                        actionText = "index"
                    else:
                        keyChanges.update({"kpA2" : ""})

                    if(middleCond):
                        keyChanges.update({"kpA3" : actionDict["middleAction"]})
                        actionText = "middle"
                    else:
                        keyChanges.update({"kpA3" : ""})

                    if(ringCond):
                        keyChanges.update({"kpA4" : actionDict["ringAction"]})
                        actionText = "ring"
                    else:
                        keyChanges.update({"kpA4" : ""})

                    if(pinkyCond):
                        keyChanges.update({"kpA5" : actionDict["pinkyAction"]})
                        actionText = "pinky"
                    else:
                        keyChanges.update({"kpA5" : ""})
                
                handleKeyChanges(keyChanges)

                cv2.putText(image, actionText, (380, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def addHeadDetectionImage(image, movementText, angleText, keypressX, keypressY, nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix, nose_2d):
        # Add the movementText on the image
        cv2.putText(image, movementText, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, angleText, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, str(keypressX) + " | " + str(keypressY), (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
        # Display the nose direction
        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
        
        # cv2.line(image, p1, p2, (255, 0, 0), 2)

    # Web Cam Input
    cap = cv2.VideoCapture(1) #Set 1 for other cam, 0 for default cam
    # Set 1920x1080
    cap.set(3, 640)
    cap.set(4, 384)

    # Mediepipe Meshes
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5,max_num_hands=1)

    print("Set up complete")

    while cap.isOpened():

        # Mini Delay
        sleepTime = 0.025
        time.sleep(sleepTime)

        success, image = cap.read()

        #Flip the image horizontally for a later selfie-view display
        #Also convert the color spacce from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Get the results
        faceResults = face_mesh.process(image)
        handsResults = hands.process(image)

        # To improve performance
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        imageHeight, imageWidth, _ = image.shape
        
        if(not pauseDetection):
            # Initiate Hand Gesture Estimation
            runHandGestureEstimation()

            # Initiate Head Pose Estimation
            runHeadPoseEstimation()
        
        cv2.imshow('Detection Estimation', image)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or cv2.getWindowProperty('Detection Estimation', cv2.WND_PROP_VISIBLE) <1:
            break
        
    cv2.destroyAllWindows()
    cap.release()

def initGUI():
    global defaultHeadPoseCoords
    global headPoseCoords

    def actionDetection():
        global pauseDetection
        pauseDetection = not pauseDetection
        if(actionDetectButton["text"] == "Stop Action"):
            actionDetectButton["text"] = "Start Action"
            actionDetectButton["bg"] = "green"
        else:
            actionDetectButton["text"] = "Stop Action"
            actionDetectButton["bg"] = "red"

    def revertDefaultCoords():
        for key, value in defaultHeadPoseCoords.items():
           currEntry = coordEntryDict[key]
           currEntry.delete(0, 'end')
           currEntry.insert(0, value)
           headPoseCoords[key]=value

    def updateCoords():
        for key,value in coordEntryDict.items():
            headPoseCoords[key]=float(value.get())

    def selectGameProfile(*args):
        global actionDict

        selectedProfile = ap[selectedGame.get()]

        for key,value in actionDict.items():
            actionDict[key] = selectedProfile[key]

        print(selectedGame.get() + " has been selected, Actions updated")

    def createThreadForCam():
        initDetectionButton["text"] = "Initializing... DO NOT click anything else untill detection appears"
        initDetectionButton["bg"] = "black"
        initDetectionButton["fg"] = "red"
        initDetectionButton["state"] = "disabled"
        CamThread = Thread(target=initCamDetection)
        CamThread.setDaemon(True)
        CamThread.start()

    root = tk.Tk()
    root.title("THIAN'S SCUFFED FACE HAND GAMING CONTROLLER")
    
    title = tk.Label(
        root, 
        text="THIAN'S SCUFFED FACE HAND \n GAMING CONTROLLER",
        font = ("Calibri", 16, "bold"),
    )
    title.grid(row = 0, column = 0, pady = 2, columnspan = 2)

    gameProfileLabel = tk.Label(
        root, 
        text="Select your Game",
        font = ("Calibri", 12),
    )
    gameProfileLabel.grid(row = 1, column = 0, pady = 2)

    selectedGame = tk.StringVar()
    selectedGame.set("DEFAULT")
    selectedGame.trace('w', selectGameProfile)

    gameDropdown = tk.OptionMenu(
        root,
        selectedGame,
        *list(ap)
    )
    gameDropdown.grid(row = 1, column = 1, sticky = "w", pady = 2)

    initDetectionButton = tk.Button(
        root,
        text="Initiate Camera Detection",
        font=("Calibri", 12, "bold"),
        wraplength=250,
        width=30,
        height=2,
        bg="blue",
        fg="white",
        command=createThreadForCam
    )
    initDetectionButton.grid(row = 2, column = 0, pady = 2, columnspan = 2)

    actionDetectButton = tk.Button(
        root,
        text="Stop Action",
        font=("Calibri", 12, "bold"),
        width=18,
        height=2,
        bg="red",
        fg="white",
        command=actionDetection
    )
    actionDetectButton.grid(row = 3, column = 0, pady = 2, columnspan = 2)

    coordTitle = tk.Label(
        root,
        text="Head Pose Coordinates",
        font=("Calibri", 14, "bold"),
    )
    coordTitle.grid(row = 4, column = 0, pady = 2, columnspan = 2)

    # Create coords labels, entries and buttons
    coordRow = 5

    coordEntryDict = {}
    for key, value in defaultHeadPoseCoords.items():
        coordLabel = tk.Label(
            root,
            text=key + ":"
        )
        coordLabel.grid(row = coordRow, column = 0, pady = 2)
        
        coordEntry = tk.Entry()
        coordEntry.insert(0, value)
        coordEntry.grid(row = coordRow, column = 1, pady = 2)

        coordEntryDict[key] = coordEntry

        coordRow += 1

    updateButton = tk.Button(
        root,
        text="Update Coordinates",
        font=("Calibri", 12, "bold"),
        width=18,
        height=3,
        bg="yellow",
        fg="black",
        command=updateCoords
    )
    updateButton.grid(row = coordRow + 1, column = 0, pady = 2)

    defaultCoordButton = tk.Button(
        root,
        text="Revert Coordinates",
        font=("Calibri", 12, "bold"),
        width=18,
        height=3,
        bg="grey",
        fg="white",
        command=revertDefaultCoords
    )
    defaultCoordButton.grid(row = coordRow + 1, column = 1, pady = 2)

    quitButton = tk.Button(
        root,
        text="Quit",
        font=("Calibri", 12, "bold"),
        width=18,
        height=2,
        bg="green",
        fg="white",
        command=quit
    )
    quitButton.grid(row = coordRow + 2, column = 0, pady = 2, columnspan = 2)

    print("GUI Launched")

    root.mainloop()

# Global Variables
pauseDetection = False
movementText = ""
actionText = ""
keyPressDict = {
    "kpX" : "",
    "kpY" : "",
    "kpA1" : "",
    "kpA2" : "",
    "kpA3" : "",
    "kpA4" : "",
    "kpA5" : "",
    "kpA6" : ""}
actionDict = {
    "upMove" : "",
    "downMove" : "",
    "leftMove": "",
    "rightMove": "",
    "thumbAction" : "",
    "indexAction" : "",
    "middleAction" : "",
    "ringAction" : "",
    "pinkyAction" : "",
    "fistAction": ""}
headPoseCoords = {
    "topLeftX" : -8,
    "topLeftY" : 10,
    "btmLeftX" : -4,
    "btmLeftY" : 2,
    "topRightX" : 9,
    "topRightY" : 13,
    "btmRightX" : 8,
    "btmRightY" : 5,
    "leftX" : -6,
    "rightX" : 8,
    "downY" : 1,
    "upY" : 8
}
defaultHeadPoseCoords = {
    "topLeftX" : -8,
    "topLeftY" : 10,
    "btmLeftX" : -4,
    "btmLeftY" : 2,
    "topRightX" : 9,
    "topRightY" : 13,
    "btmRightX" : 8,
    "btmRightY" : 2,
    "leftX" : -6,
    "rightX" : 8,
    "downY" : 1,
    "upY" : 8
}

# Initialise Exit Handler, Key press controllers
atexit.register(exit_handler)
keyboard = KeyboardController()
mouse = MouseController()

# Initialise Graphics Use Interface
initGUI()