import cv2
import json

image_path = "C:\\Users\\Paromita Roy\\OneDrive\\Documents\\Coursework\\Capstone\\SER517_Group35_Capstone\\ProDataset\\test\\images\\00010__00504.jpg"

metadata = {"device": "iPad Air 2", "screen_h": 1024, "screen_w": 768, "face_valid": 1, "face_x": 163, "face_y": 415, "face_w": 222, "face_h": 222, "leye_x": 281, "leye_y": 422, "leye_w": 67, "leye_h": 67, "reye_x": 194, "reye_y": 422, "reye_w": 67, "reye_h": 67, "dot_xcam": -3.2197756857, "dot_y_cam": -10.2233384972, "dot_x_pix": 223.151406765, "dot_y_pix": 478.263533115, "reye_x1": 210, "reye_y1": 465, "reye_x2": 246, "reye_y2": 461, "leye_x1": 302, "leye_y1": 459, "leye_x2": 338, "leye_y2": 460}
img = cv2.imread(image_path)

print(img)

# Extract left and right eye regions using the coordinates
leye_roi = img[metadata['leye_y']:metadata['leye_y']+metadata['leye_h'], metadata['leye_x']:metadata['leye_x']+metadata['leye_w']]
reye_roi = img[metadata['reye_y']:metadata['reye_y']+metadata['reye_h'], metadata['reye_x']:metadata['reye_x']+metadata['reye_w']]

print("lefteye roi", len(leye_roi[0]))
print("righteye roi", len(reye_roi))

# Resize images to 128x128
leye_resized = cv2.resize(leye_roi, (128, 128))
reye_resized = cv2.resize(reye_roi, (128, 128))

# Save the cropped images
cv2.imwrite('new_left_eye.jpg', leye_resized)
cv2.imwrite('new_right_eye.jpg', reye_resized)