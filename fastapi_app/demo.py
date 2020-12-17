from model import MaskValidator
import cv2

cap = cv2.VideoCapture(0)
model = MaskValidator()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    result = model.validate_mask(frame)

    # Add label to the display if there is result from the validator
    if result is not None:
        label = result['label']
        startX = result['startX']
        startY = result['startY']
        endX = result['endX']
        endY = result['endY']
        color = (0, 255, 0) if label == "Correct" else (0, 0, 255)

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
