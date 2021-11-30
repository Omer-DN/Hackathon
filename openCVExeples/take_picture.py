import cv2

camera = cv2.VideoCapture(0)
for i in range(20):
    return_value, image = camera.read()
    cv2.imwrite('my_picture'+str(i) + '.jpg', image)

for i in range(20):
    img = cv2.imread(r'C:\networks\work\openCV\my_picture' + str(i)+'.jpg')
    cv2.imshow('Input', img)
    cv2.waitKey(0)

del(camera)