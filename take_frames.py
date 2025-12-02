import cv2
import os

cap = cv2.VideoCapture(0)
count = 0
folder = "/Users/bhumikaburhade/Desktop/dataset/human"  #"/Users/bhumikaburhade/Desktop/dataset/human"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    
    if key == ord('s'):
        cv2.imwrite(f"{folder}/frame_{count}.jpg", frame)
        count += 1
        print(f"Saved frame {count}")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()