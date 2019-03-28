 # import the necessary packages
import numpy as np
import imutils
import glob
import cv2
import crop_rotate as cr


if __name__ == "__main__":

    # if v == True see the various step of scaling and rotation
    v = False
    # different time of algo for matching, work better with true
    using_min = True
    # text for images
    templates_text = ["Normal", "Flipped"]
    rotation_text = ["Normal", "Right", "Left"]
    # canny threshold values
    canny_min = 50
    canny_max = 100
    canny_win = 7

    # load the image image, convert it to grayscale, and detect edges
    template = cv2.imread("/Users/leonardotanzi/Desktop/MasterThesis/Templates/Bacino_left.jpg")
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, canny_min, canny_max, canny_win)
    # flip the template
    templates = [template, cv2.flip(template, 1)]
    # get template size
    (tH, tW) = template.shape[:2]

    # loop over the images to find the template in
    for imagePath in glob.glob("/Users/leonardotanzi/Desktop/TesiMagistrale/Audisio/ImmaginiBacinoCopiaCrop/*.jpg"):

        for i in range(2):

            image = cv2.imread(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            found = None

            # loop over the scales of the image
            for scale in np.linspace(0.9, 1.1, 5)[::-1]:
                # linspace(start, stop, num of samples to gen) the - 1 reverse the list So, when you do a[::-1] , it
                # starts from the end, towards the first, taking each element. So it reverses a resize the image
                # according to the scale, and keep track of the ratio of the resizing
                resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
                r = gray.shape[1] / float(resized.shape[1])

                # if the resized image is smaller than the template, then break from the loop
                if resized.shape[0] < tH or resized.shape[1] < tW:
                    break
                # list with original image, image rotated 5 degrees and image rotated -5 degrees
                images = [resized, cr.crop_rotate(resized, 5), cr.crop_rotate(resized, -5)]

                j = 0
                for img in images:
                    # detect edges in the resized, grayscale image and apply template
                    # matching to find the template in the image
                    edged = cv2.Canny(img, canny_min, canny_max, canny_win)

                    if using_min:
                        result = cv2.matchTemplate(edged, template[i], cv2.TM_SQDIFF)
                        val, _, loc, _ = cv2.minMaxLoc(result)
                    else:
                        result = cv2.matchTemplate(edged, template[i], cv2.TM_CCOEFF)
                        _, val, _, loc = cv2.minMaxLoc(result)

                    # check to see if the iteration should be visualized
                    if v:
                        # draw a bounding box around the detected region
                        clone = np.dstack([edged, edged, edged])
                        cv2.rectangle(clone, (loc[0], loc[1]), (loc[0] + tW, loc[1] + tH), (0, 0, 255), 30)
                        cv2.imshow("Image {}, Template {}, Scale {}, Rotation {}".format(imagePath.split("/")[-1],
                                                                                        templates_text[i],
                                                                                        scale,
                                                                                        rotation_text[j]),
                                   cv2.resize(clone, (500, 500)))
                        cv2.waitKey(0)

                    # if we have found a new maximum/minimum correlation value, then update the bookkeeping variable
                    # weight the val if left or right
                    if i == 0:  # left
                        # update the min/max if it's the first cycle or if the weighted val is less/more than the previo
                        # weight the val using the position, in the left case, the more the point is at left,
                        # the more is weighted (so if loc[0] -> 0 (left side) the value must increase,
                        # if loc[0]->infinite (right side) the value must decrease

                        # più è a sinistra, quindi piu loc[0] è piccolo, più è piccolo val * loc[0], perchè sto set min
                        if (found is None or val * loc[0] < found[0]) and using_min is True:
                            found = (val, loc, r)
                        # più è a sinistra, quindi piu loc[0] è piccolo, più è grande val * loc[0], perchè sto set max
                        elif (found is None or (val / (loc[0] if loc[0] != 0 else 0.01)) > found[0]) and using_min is False:
                            found = (val, loc, r)
                    else:  # right
                        # più è a destra, quindi piu loc[0] è grande, più è piccolo val * loc[0], perchè sto set min
                        if (found is None or (val / (loc[0] if loc[0] != 0 else 0.01)) < found[0]) and using_min is True:
                            found = (val, loc, r)
                        # più è a destra, quindi piu loc[0] è grande, più è grande val * loc[0], perchè sto set max
                        elif (found is None or val * loc[0] > found[0]) and using_min is False:
                            found = (val, loc, r)
                    # print("Value is {}".format(val))
                    j += 1

            # unpack the bookkeeping variable and compute the (x, y) coordinates
            # of the bounding box based on the resized ratio
            (_, loc, r) = found
            (startX, startY) = (int(loc[0] * r), int(loc[1] * r))
            (endX, endY) = (int((loc[0] + tW) * r), int((loc[1] + tH) * r))

            # draw a bounding box around the detected result and display the image
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 30)
            cv2.imshow("Image", cv2.resize(image, (500, 500)))
            cv2.waitKey(0)



