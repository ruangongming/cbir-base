import cv2

I=cv2.imread("D:\\HocTapEPU\\CBIR\\CBIRThi\\static\\img\\img1.jpg")

I_r = cv2.resize(I,(120,120))

cv2.imwrite("filename.jpg", I_r)


