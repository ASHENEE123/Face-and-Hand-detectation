import cv2
import mediapipe as mp
import pyttsx3 
mp_hands=mp.solutions.hands
hands=mp_hands.Hands()
engine=pyttsx3.init()
mp_mesh=mp.solutions.face_mesh
mp_draw=mp.solutions.drawing_utils
mp_spec=mp.solutions.drawing_styles
spec=mp_draw.DrawingSpec(thickness=1,circle_radius=1)
img1=cv2.VideoCapture(0)
with mp_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
  while img1.isOpened():
    bo,img=img1.read()
    res=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=hands.process(res)
    result1=face_mesh.process(res)
    if result.multi_hand_landmarks:
        for hand_marks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img,hand_marks,mp_hands.HAND_CONNECTIONS)
            print(hand_marks)
    if  result1.multi_face_landmarks:
          for face in result1.multi_face_landmarks:
               mp_draw.draw_landmarks(img,face,mp_mesh.FACEMESH_TESSELATION,mp_spec.get_default_face_mesh_tesselation_style())
          
    cv2.imshow("img",cv2.flip(img,1))
    if cv2.waitKey(5) & 0XFF==ord('q'):
        break
engine.say("face and hands detected correctly")
engine.runAndWait()
img1.release()
cv2.destroyAllWindows()