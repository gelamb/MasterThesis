# import the necessary packages
import numpy as np
import imutils
import glob
import cv2
import crop_rotate as cr
from PIL import Image


def pre_process_image(image):
    canny_win = 17  # 27 with dark images like 118 and 138
    canny_aperture = 3
    gaussian_win = 3
    blur = cv2.GaussianBlur(image, (gaussian_win, gaussian_win), 0)
    edged = cv2.Canny(blur, canny_win, canny_win * 3, canny_aperture)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    eroded = cv2.erode(edged, kernel_erode, iterations=2)
    dilated = cv2.dilate(eroded, kernel_dilate, iterations=2)
    return dilated


def print_img(name, image, save=False, save_name=None):

    if save is False:
        cv2.namedWindow(name)
        cv2.moveWindow(name, 370, 140)
        scale_percent = 20  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        cv2.imshow(name, cv2.resize(image, dim))
        cv2.waitKey(0)
        cv2.destroyWindow(name)
    else:
        cv2.imwrite(name + "{}.png".format(save_name), image)


if __name__ == "__main__":

    # if v == True see the various step of scaling and rotation
    v = False
    # different time of algo for matching, work better with true
    using_min = True
    using_rotation = True
    using_scale = 1  # if 1 not scale and scale_index must be 0, if > 1 number of scaled images
    scale_index = 0 if using_scale is 1 else 0.1
    using_weights = True
    save_images = False
    alternative_approach = True
    # text for images
    templates_text = ["Normal", "Flipped"]
    rotation_text = ["Normal", "Right", "Left"]
    output_path = "/Users/leonardotanzi/Desktop/MasterThesis/Output/"
    # canny threshold values

    # load the image image, convert it to grayscale, and detect edges
    if alternative_approach:
        template = cv2.imread("/Users/leonardotanzi/Desktop/MasterThesis/Templates/Bacinofake.jpg")
    else:
        template = cv2.imread("/Users/leonardotanzi/Desktop/MasterThesis/Templates/Bacino_left3.jpg")
    template = pre_process_image(template)
    # flip the template
    templates = [template, cv2.flip(template, 1)]
    # get template size
    (tH, tW) = template.shape[:2]

    # loop over the images to find the template in
    for imagePath in glob.glob("/Users/leonardotanzi/Desktop/TesiMagistrale/Audisio/ImmaginiBacinoCopiaCrop/*.jpg"):

        if alternative_approach:
            distance_from_border = 60
            image = cv2.imread(imagePath)
            edged = pre_process_image(image)
            (h, w) = edged.shape[:2]

            if using_min:
                result = cv2.matchTemplate(edged, templates[0], cv2.TM_SQDIFF)
                val, _, loc, _ = cv2.minMaxLoc(result)
            else:
                result = cv2.matchTemplate(edged, templates[0], cv2.TM_CCOEFF)
                _, val, _, loc = cv2.minMaxLoc(result)

            clone = np.dstack([edged, edged, edged])
            # cv2.rectangle(clone, (loc[0], loc[1]), (loc[0] + tW, loc[1] + tH), (0, 0, 255), 30)
            # cv2.circle(clone, (loc[0], loc[1]), 30, (0, 255, 0), thickness=-1)
            box_left = (0 + distance_from_border, loc[1] - int(tH/4), loc[0] + int(tW/4), h - distance_from_border)
            box_right = (loc[0] + int(3/4*tW), loc[1] - int(tH/4), w - distance_from_border, h - distance_from_border)
            cv2.rectangle(clone, (box_left[0], box_left[1]), (box_left[2], box_left[3]), (0, 0, 255), 10)
            cv2.rectangle(clone, (box_right[0], box_right[1]), (box_right[2], box_right[3]), (0, 0, 255), 10)
            if v:
                print_img(imagePath.split("/")[-1], clone)
            image_to_crop = Image.open(imagePath)
            crop_image_left = image_to_crop.crop(box_left)
            crop_image_right = image_to_crop.crop(box_right)

            crop_image_left.save(output_path + ("{}_left.jpg".format(imagePath.split("/")[-1].split(".")[-2])))
            crop_image_right.save(output_path + ("{}_right.jpg".format(imagePath.split("/")[-1].split(".")[-2])))


        else:
            for i in range(2):

                image = cv2.imread(imagePath)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                found = None

                # loop over the scales of the image
                for scale in np.linspace(1 - scale_index, 1 + scale_index, using_scale)[::-1]:
                    # linspace(start, stop, num of samples to gen) the - 1 reverse the list So, when you do a[::-1] , it
                    # starts from the end, towards the first, taking each element. So it reverses a resize the image
                    # according to the scale, and keep track of the ratio of the resizing
                    resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
                    r = gray.shape[1] / float(resized.shape[1])

                    # if the resized image is smaller than the template, then break from the loop
                    if resized.shape[0] < tH or resized.shape[1] < tW:
                        break

                    if using_rotation:
                        # list with original image, image rotated 5 degrees and image rotated -5 degrees
                        images = [resized, cr.crop_rotate(resized, 5), cr.crop_rotate(resized, -5)]
                    else:
                        images = [resized]

                    j = 0
                    for img in images:
                        # just for the output text
                        if using_rotation is False:
                            j = 0
                        # detect edges in the resized, grayscale image and apply template
                        # matching to find the template in the image
                        edged = pre_process_image(img)

                        if using_min:
                            result = cv2.matchTemplate(edged, templates[i], cv2.TM_SQDIFF)
                            val, _, loc, _ = cv2.minMaxLoc(result)
                        else:
                            result = cv2.matchTemplate(edged, templates[i], cv2.TM_CCOEFF)
                            _, val, _, loc = cv2.minMaxLoc(result)

                        # check to see if the iteration should be visualized
                        if v:
                            # print_img("t1", templates[i], save=False)
                            window_name = "Image {}, Template {}, Scale {}, Rotation {}".format(imagePath.split("/")[-1],
                                                                                             templates_text[i],
                                                                                             scale,
                                                                                             rotation_text[j])
                            # draw a bounding box around the detected region
                            clone = np.dstack([edged, edged, edged])
                            cv2.rectangle(clone, (loc[0], loc[1]), (loc[0] + tW, loc[1] + tH), (0, 0, 255), 30)
                            print_img(window_name, clone)

                        # if we have found a new maximum/minimum correlation value, then update the bookkeeping variable
                        # weight the val if left or right
                        if using_weights:
                            mult = val * loc[0]
                            div = val / (loc[0] if loc[0] != 0 else 0.01)

                            if i == 0:  # left
                                # update the min/max if it's the first cycle or if the weighted val is less/more than
                                # the previous weight the val using the position, in the left case, the more the point
                                # is atleft, the more is weighted (so if loc[0] -> 0 (left side) the value must increase
                                # if loc[0]->infinite (right side) the value must decrease

                                # più è a sinistra, quindi piu loc[0] è piccolo, più è piccolo val * loc[0], perchè sto set min
                                if (found is None or mult < found[0]) and using_min is True:
                                    found = (mult, loc, r)
                                # più è a sinistra, quindi piu loc[0] è piccolo, più è grande val / loc[0], perchè sto set max
                                elif (found is None or div > found[0]) and using_min is False:
                                    found = (div, loc, r)
                            else:  # right
                                # più è a destra, quindi piu loc[0] è grande, più è piccolo val / loc[0], perchè sto set min
                                if (found is None or div < found[0]) and using_min is True:
                                    found = (div, loc, r)
                                # più è a destra, quindi piu loc[0] è grande, più è grande val * loc[0], perchè sto set max
                                elif (found is None or mult > found[0]) and using_min is False:
                                    found = (mult, loc, r)
                        else:
                            if i == 0:  # left
                                if (found is None or val < found[0]) and using_min is True:
                                    found = (val, loc, r)
                                elif (found is None or val > found[0]) and using_min is False:
                                    found = (val, loc, r)
                            else:  # right
                                if (found is None or val < found[0]) and using_min is True:
                                    found = (val, loc, r)
                                elif (found is None or val > found[0]) and using_min is False:
                                    found = (val, loc, r)

                        j += 1

                # unpack the bookkeeping variable and compute the (x, y) coordinates
                # of the bounding box based on the resized ratio
                (_, loc, r) = found
                (startX, startY) = (int(loc[0] * r), int(loc[1] * r))
                (endX, endY) = (int((loc[0] + tW) * r), int((loc[1] + tH) * r))

                # draw a bounding box around the detected result and display the image
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 30)
                name = (imagePath.split("/")[-1]).split(".")[-2] + "_" + str(i)
                print_img("TemplateDetect", image, save=True, save_name=name)

