 # import the necessary packages
import numpy as np
import imutils
import glob
import cv2
import crop_rotate


# load the image image, convert it to grayscale, and detect edges
template = cv2.imread("/Users/leonardotanzi/Desktop/MasterThesis/Templates/Bacino_left.jpg")
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 100, 7)
(tH, tW) = template.shape[:2]
template2 = cv2.resize(template, (400, 400))
cv2.imshow("Template", template2)
v = True
using_min = False

# loop over the images to find the template in
for imagePath in glob.glob("/Users/leonardotanzi/Desktop/TesiMagistrale/Audisio/ImmaginiBacinoCopiaCrop/*.jpg"):
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None

    # loop over the scales of the image
    for scale in np.linspace(0.8, 1.0, 10)[::-1]: # linspace(start, stop, num of samples to gen) the - 1 reverse the
                                                #  list So, when you do a[::-1] , it starts from the end, towards the
                                                #  first, taking each element. So it reverses a
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 100)

        if using_min:
            result = cv2.matchTemplate(edged, template, cv2.TM_SQDIFF)
            val, y, loc, _ = cv2.minMaxLoc(result)
        else:
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            y, val, _, loc = cv2.minMaxLoc(result)

        # check to see if the iteration should be visualized
        if v:
            # draw a bounding box around the detected region
            clone = np.dstack([edged, edged, edged])
            cv2.rectangle(clone, (loc[0], loc[1]), (loc[0] + tW, loc[1] + tH), (0, 0, 255), 30)
            cv2.imshow("Visualize", cv2.resize(clone, (500, 500)))
            cv2.waitKey(0)

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if (found is None or val < found[0]) and using_min is True:
            found = (val, loc, r)
        elif (found is None or val > found[0]) and using_min is False:
            found = (val, loc, r)

        print("Value is {} and y {}".format(val, y))
    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, loc, r) = found
    (startX, startY) = (int(loc[0] * r), int(loc[1] * r))
    (endX, endY) = (int((loc[0] + tW) * r), int((loc[1] + tH) * r))

    print("Minimum value is {}".format(found[0]))

    # draw a bounding box around the detected result and display the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 30)
    cv2.imshow("Image", cv2.resize(image, (500, 500)))
    cv2.waitKey(0)



