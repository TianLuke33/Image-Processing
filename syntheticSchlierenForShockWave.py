############
# Image Processing for Synthetic Schlieren
# Synthetic schlieren is a technique for gas or transparent material's images. This code was used for a shock wave in air. 
# Input: Two folder of  png images. 'images' has test images; 'FLATS' has background images
# Output: A video to show the result
# Author: Tianluke33
# Date: 7/30/2024
import cv2
import os

image_folder = 'images'
video_name = 'processedImage3.avi'
fps = 10
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video = cv2.VideoWriter(video_name, fourcc, fps, (200,1250))
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
background = cv2.imread(os.path.join('FLATS', images[4]))
index = 0
count = 1
meanframe =cv2.absdiff(cv2.imread(os.path.join(image_folder, images[0])),background)
while index < len(images):
    frame = cv2.imread(os.path.join(image_folder, images[index]))
    #calculate the difference between current img and background
    frame0 = cv2.absdiff(frame,background)

    # contrast gain, remove pixels low values(<20) and high values(>60)
    ret, frame1 = cv2.threshold(frame0,20,255,cv2.THRESH_TOZERO)
    ret, frame1 = cv2.threshold(frame0,60,255,cv2.THRESH_TOZERO_INV)
    frame1 = frame1*4

    # try different filters
    #frame2 = cv2.blur(frame1,(5,5))
    #frame3 = cv2.GaussianBlur(frame1,(5,5),0)
    #frame4 = cv2.medianBlur(frame1,5)
    meanframe =cv2.addWeighted(meanframe,1-1/count,frame,1/count,0)
    count += 1
    frame5 = cv2.absdiff(frame1,meanframe)
    # frame5 = cv2.fastNlMeansDenoising(frame, 100,200) doesn't work very well
    frame2 = cv2.blur(frame5,(5,5))
    frame3 = cv2.GaussianBlur(frame5,(5,5),0)
    frame4 = cv2.medianBlur(frame5,5)

    # put every frame together
    final_images = cv2.vconcat([frame,frame2,frame3,frame4,frame5])
    #height, width, layers = final_images.shape # Need to check the image size fro writing video
    # write result into a video
    video.write(final_images)
    # show the result
    cv2.imshow('final',final_images)
    background = frame
    
    index += 1
    k = cv2.waitKey(33)
    if k==27: # Esc key to stop
        break

video.release()
cv2.destroyAllWindows()
