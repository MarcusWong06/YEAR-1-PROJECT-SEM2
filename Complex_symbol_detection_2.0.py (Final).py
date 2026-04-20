from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time

sample_dict= {0:"hazardSymbol.jpg", 1:"fingerPrint.jpg",2:"recycleSymbol.jpg",3:"qrCode.jpg", 4:"pushButton.jpg"}


'''
0) OK (48)
1) OK (50)
2) OK (35)
3) OK (43)
4) OK (60)
'''

def func_detect_complex_img(picam2,dict_key):
    img_reference = cv.imread(sample_dict[dict_key], cv.IMREAD_GRAYSCALE)
    if img_reference is None:
        raise FileNotFoundError(f"{sample_dict[dict_key]} not found")
    
    orb = cv.ORB_create(nfeatures=1200)
    kp1, des1 = orb.detectAndCompute(img_reference, None)


    if(dict_key == 2):
        counter = 8
    else:
        counter = 5
        
    num_good = 0
    for _ in range(counter):
        frame = picam2.capture_array()
        img_scene = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        kp2, des2 = orb.detectAndCompute(img_scene, None)
        if des1 is None or des2 is None:
            return None

        ''' Using Lowe's ratio test to filter matches '''
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        img_matches = cv.drawMatches(
            img_reference, kp1,
            img_scene, kp2,
            good_matches, None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv.imshow("Display", img_matches)
        cv.waitKey(1)

        num_good += len(good_matches)
    
    # ---- Accuracy Calculation ----
    num_good = round(num_good / float(counter))
    return num_good



def main():
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"size": (640, 480)}
        )
    )
    picam2.start()

    try:
        while True:
            img_found = False
            frame = picam2.capture_array()
            BGR_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            
            for i in range(len(sample_dict)):
                result = func_detect_complex_img(picam2,i)
                if result is None:
                    cv.imshow("Camera Feed", BGR_frame)
                    cv.waitKey(1)
                    break

                if (result >= 48) and (i == 0):
                    print(f"Hazard Symbol Detected!")
                    img_found = True
                    break
                elif (result >= 50) and (i == 1):
                    print(f"Fingerprint Detected!")
                    img_found = True
                    break
                elif (result >= 35) and (i == 2):
                    print(f"Recycle Symbol Detected!")
                    img_found = True
                    break
                elif (result >= 43) and (i == 3):
                    print(f"QR Code Detected!")
                    img_found = True
                    break
                elif (result >= 60) and (i == 4):
                    print(f"Push Button Detected!")
                    img_found = True
                    break
                else:
                    continue                 
            
            if(not img_found):
                print("No Image Detected!")
                cv.imshow("Display", BGR_frame)
                cv.waitKey(1)
            


            if cv.waitKey(1) & 0xFF == 27:
                break
    finally:
        cv.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
