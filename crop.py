import cv2 

def crop(image: cv2.typing.MatLike):
    structure = cv2.getStructuringElement( 
                              shape = cv2.MORPH_RECT, ksize =(22, 3)) 

    grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(grayed, (5, 5), 0)

    sobelx = cv2.Sobel(blured, cv2.CV_8U, 1, 0, ksize = 3)  
        
    ret2, threshold_img = cv2.threshold(sobelx, 0, 255, 
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU) 

    element = structure
    threshhold_copy = threshold_img.copy() 
    cv2.morphologyEx(src = threshold_img, 
                        op = cv2.MORPH_CLOSE, 
                        kernel = element, 
                        dst = threshhold_copy) 


    # Getting contours

    contours, _ = cv2.findContours(threshhold_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    countoured_image = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv2.imshow("countoured_image", countoured_image)


    return grayed 
