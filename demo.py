# # import cv2
# # import tensorflow as tf
# # from sudoku_images_loader import SudokuDataLoader
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import os
# # from imutils import contours

# # class ImageProcessing():

# #     def plotting(self, images, title):
# #         plt.imshow(images,'gray')
# #         plt.title(title)
# #         plt.xticks([]),plt.yticks([])
# #         plt.show()

# #     def blurring(self, X_train, kernel_size ):
# #         return cv2.GaussianBlur(X_train,kernel_size,0)

# #     def adaptive_thresholding(self, X_train):
# #         return cv2.adaptiveThreshold(X_train, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 5, 2)

# #     def invertion(self,X_train):
# #         return cv2.bitwise_not(X_train)

# #     def dilation(self, X_train, kernel):
# #         return cv2.dilate(X_train, kernel)

# #     def erosion(self, X_train, kernel):
# #         return cv2.erode(X_train, kernel, iterations=1) 

# #     def detect_grid(self, X_train):
# #         ##cv2.floodFill(image, mask, seedPoint, newVal [, loDiff[, upDiff[, flags]]] )
# #         return X_train

# #     def show_image(self, img):
# #         """Shows an image until any key is pressed"""
# #         print(type(img))
# #         print(img.shape)
# #         cv2.imshow('image', img)  # Display the image
# #         cv2.imwrite('images/gau_sudoku3.jpg', img)
# #         cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
# #         cv2.destroyAllWindows()  # Close all windows
# #         return img

# #     def convert_when_colour(self, colour, img):
# #         """Dynamically converts an image to colour if the input colour is a tuple and the image is grayscale."""
# #         if len(colour) == 3:
# #             if len(img.shape) == 2:
# #                 img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# #             elif img.shape[2] == 1:
# #                 img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# #         return img

# #     def detect_cells(self, X_train,colour=(0, 0, 255), thickness=2):
# #         # X_train = np.array(X_train)
# #         # print(type(X_train))
# #         # X_train = cv2.cvtColor(cv2.UMat(X_train), cv2.COLOR_BGR2GRAY)
# #         # X_train = cv2.adaptiveThreshold(X_train,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,57,5)

# #         # contours, hierarchy = cv2.findContours(X_train, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# #         # img = cv2.drawContours(X_train, contours, -1, (0,255,0), 3)
# #         # print(type(img))

# #         """Finds the 4 extreme corners of the largest contour in the image."""
# #         opencv_version = cv2.__version__.split('.')[0]
# #         if opencv_version == '3':
# #             _, contours, h = cv2.findContours(X_train.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
# #         else:
# #             contours, h = cv2.findContours(X_train.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
# #         contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
# #         polygon = contours[0]  # Largest image
        
# #         """Displays contours on the image."""
# #         img = convert_when_colour(colour, X_train.copy())
# #         img = cv2.drawContours(img, contours, -1, colour, thickness)
# #         show_image(img)

# #         #plt.imshow("", X_train)
        
# # if __name__=='__main__':

# #     DATA_DIR = "dataset/sudoku"
# #     DATA_TYPE = ["train","test"]
# #     # sudokudataloader = SudokuDataLoader(DATA_DIR, DATA_TYPE)
# #     # X_train, X_test = sudokudataloader.load_data()
# #     # sudokudataloader.load_data()
# #     # print(X_train.shape)

# #     X_train = cv2.imread("sample.jpg")
# #     imageprocessing = ImageProcessing()

# #    # imageprocessing.plotting(X_train, "Original")

# #     # kernel_size = (9,9)
# #     # X_train = imageprocessing.blurring(X_train, kernel_size)
# #     # # imageprocessing.plotting(X_train, "Blurred")

# #     # X_train = imageprocessing.adaptive_thresholding(X_train)
# #     # # imageprocessing.plotting(X_train, "Thresholding")

    
# #     # X_train = imageprocessing.invertion(X_train)
# #     # # imageprocessing.plotting(X_train, "Invertion")

# #     # kernel_size = (5,5)
# #     # kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
# #     # X_train = imageprocessing.dilation(X_train,kernel)
# #     # # imageprocessing.plotting(X_train, "Dilation")

# #     # X_train = imageprocessing.erosion(X_train,kernel)
# #     # imageprocessing.plotting(X_train, "Erosion")

# #     imageprocessing.detect_cells(X_train)
# #     #imageprocessing.plotting(X_train, "Cell EXtraction")


# import numpy as np
# import cv2
# import operator
# import numpy as np
# import matplotlib.pyplot as plt
# import sys

# def plot_many_images(images, titles, rows=1, columns=2):
# 	"""Plots each image in a given list as a grid structure. using Matplotlib."""
# 	for i, image in enumerate(images):
# 		plt.subplot(rows, columns, i+1)
# 		plt.imshow(image, 'gray')
# 		plt.title(titles[i])
# 		plt.xticks([]), plt.yticks([])  # Hide tick marks
# 	plt.show()


# def show_image(img):
#     """Shows an image until any key is pressed"""
# #    print(type(img))
# #    print(img.shape)
# #    cv2.imshow('image', img)  # Display the image
# #    cv2.imwrite('images/gau_sudoku3.jpg', img)
# #    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
# #    cv2.destroyAllWindows()  # Close all windows
#     return img


# def show_digits(digits, colour=255):
#     """Shows list of 81 extracted digits in a grid format"""
#     rows = []
#     with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
#     for i in range(9):
#         row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
#         rows.append(row)
#     img = show_image(np.concatenate(rows))
#     return img
 

# def convert_when_colour(colour, img):
# 	"""Dynamically converts an image to colour if the input colour is a tuple and the image is grayscale."""
# 	if len(colour) == 3:
# 		if len(img.shape) == 2:
# 			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# 		elif img.shape[2] == 1:
# 			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# 	return img


# def display_points(in_img, points, radius=5, colour=(0, 0, 255)):
# 	"""Draws circular points on an image."""
# 	img = in_img.copy()

# 	# Dynamically change to a colour image if necessary
# 	if len(colour) == 3:
# 		if len(img.shape) == 2:
# 			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# 		elif img.shape[2] == 1:
# 			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 	for point in points:
# 		img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
# 	show_image(img)
# 	return img


# def display_rects(in_img, rects, colour=(0, 0, 255)):
# 	"""Displays rectangles on the image."""
# 	img = convert_when_colour(colour, in_img.copy())
# 	for rect in rects:
# 		img = cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour)
# 	show_image(img)
# 	return img


# def display_contours(in_img, contours, colour=(0, 0, 255), thickness=2):
# 	"""Displays contours on the image."""
# 	img = convert_when_colour(colour, in_img.copy())
# 	img = cv2.drawContours(img, contours, -1, colour, thickness)
# 	show_image(img)


# def pre_process_image(img, skip_dilate=False):
# 	"""Uses a blurring function, adaptive thresholding and dilation to expose the main features of an image."""

# 	# Gaussian blur with a kernal size (height, width) of 9.
# 	# Note that kernal sizes must be positive and odd and the kernel must be square.
# 	proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)

# 	# Adaptive threshold using 11 nearest neighbour pixels
# 	proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 	# Invert colours, so gridlines have non-zero pixel values.
# 	# Necessary to dilate the image, otherwise will look like erosion instead.
# 	proc = cv2.bitwise_not(proc, proc)

# 	if not skip_dilate:
# 		# Dilate the image to increase the size of the grid lines.
# 		kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
# 		proc = cv2.dilate(proc, kernel)

# 	return proc


# def find_corners_of_largest_polygon(img):
# 	"""Finds the 4 extreme corners of the largest contour in the image."""
# 	opencv_version = cv2.__version__.split('.')[0]
# 	if opencv_version == '3':
# 		_, contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
# 	else:
# 		contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
# 	contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
# 	polygon = contours[0]  # Largest image

# 	# Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
# 	# Each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively.

# 	# Bottom-right point has the largest (x + y) value
# 	# Top-left has point smallest (x + y) value
# 	# Bottom-left point has smallest (x - y) value
# 	# Top-right point has largest (x - y) value
# 	bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
# 	top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
# 	bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
# 	top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

# 	# Return an array of all 4 points using the indices
# 	# Each point is in its own array of one coordinate
# 	return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


# def distance_between(p1, p2):
# 	"""Returns the scalar distance between two points"""
# 	a = p2[0] - p1[0]
# 	b = p2[1] - p1[1]
# 	return np.sqrt((a ** 2) + (b ** 2))


# def crop_and_warp(img, crop_rect):
# 	"""Crops and warps a rectangular section from an image into a square of similar size."""

# 	# Rectangle described by top left, top right, bottom right and bottom left points
# 	top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

# 	# Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
# 	src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

# 	# Get the longest side in the rectangle
# 	side = max([
# 		distance_between(bottom_right, top_right),
# 		distance_between(top_left, bottom_left),
# 		distance_between(bottom_right, bottom_left),
# 		distance_between(top_left, top_right)
# 	])

# 	# Describe a square with side of the calculated length, this is the new perspective we want to warp to
# 	dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

# 	# Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
# 	m = cv2.getPerspectiveTransform(src, dst)

# 	# Performs the transformation on the original image
# 	return cv2.warpPerspective(img, m, (int(side), int(side)))


# def infer_grid(img):
# 	"""Infers 81 cell grid from a square image."""
# 	squares = []
# 	side = img.shape[:1]
# 	side = side[0] / 9

# 	# Note that we swap j and i here so the rectangles are stored in the list reading left-right instead of top-down.
# 	for j in range(9):
# 		for i in range(9):
# 			p1 = (i * side, j * side)  # Top left corner of a bounding box
# 			p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
# 			squares.append((p1, p2))
# 	return squares


# def cut_from_rect(img, rect):
# 	"""Cuts a rectangle from an image using the top left and bottom right points."""
# 	return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


# def scale_and_centre(img, size, margin=0, background=0):
# 	"""Scales and centres an image onto a new background square."""
# 	h, w = img.shape[:2]

# 	def centre_pad(length):
# 		"""Handles centering for a given length that may be odd or even."""
# 		if length % 2 == 0:
# 			side1 = int((size - length) / 2)
# 			side2 = side1
# 		else:
# 			side1 = int((size - length) / 2)
# 			side2 = side1 + 1
# 		return side1, side2

# 	def scale(r, x):
# 		return int(r * x)

# 	if h > w:
# 		t_pad = int(margin / 2)
# 		b_pad = t_pad
# 		ratio = (size - margin) / h
# 		w, h = scale(ratio, w), scale(ratio, h)
# 		l_pad, r_pad = centre_pad(w)
# 	else:
# 		l_pad = int(margin / 2)
# 		r_pad = l_pad
# 		ratio = (size - margin) / w
# 		w, h = scale(ratio, w), scale(ratio, h)
# 		t_pad, b_pad = centre_pad(h)

# 	img = cv2.resize(img, (w, h))
# 	img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
# 	return cv2.resize(img, (size, size))


# def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
# 	"""
# 	Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
# 	connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
# 	"""
# 	img = inp_img.copy()  # Copy the image, leaving the original untouched
# 	height, width = img.shape[:2]

# 	max_area = 0
# 	seed_point = (None, None)

# 	if scan_tl is None:
# 		scan_tl = [0, 0]

# 	if scan_br is None:
# 		scan_br = [width, height]

# 	# Loop through the image
# 	for x in range(scan_tl[0], scan_br[0]):
# 		for y in range(scan_tl[1], scan_br[1]):
# 			# Only operate on light or white squares
# 			if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
# 				area = cv2.floodFill(img, None, (x, y), 64)
# 				if area[0] > max_area:  # Gets the maximum bound area which should be the grid
# 					max_area = area[0]
# 					seed_point = (x, y)

# 	# Colour everything grey (compensates for features outside of our middle scanning range
# 	for x in range(width):
# 		for y in range(height):
# 			if img.item(y, x) == 255 and x < width and y < height:
# 				cv2.floodFill(img, None, (x, y), 64)

# 	mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

# 	# Highlight the main feature
# 	if all([p is not None for p in seed_point]):
# 		cv2.floodFill(img, mask, seed_point, 255)

# 	top, bottom, left, right = height, 0, width, 0

# 	for x in range(width):
# 		for y in range(height):
# 			if img.item(y, x) == 64:  # Hide anything that isn't the main feature
# 				cv2.floodFill(img, mask, (x, y), 0)

# 			# Find the bounding parameters
# 			if img.item(y, x) == 255:
# 				top = y if y < top else top
# 				bottom = y if y > bottom else bottom
# 				left = x if x < left else left
# 				right = x if x > right else right

# 	bbox = [[left, top], [right, bottom]]
# 	return img, np.array(bbox, dtype='float32'), seed_point


# def extract_digit(img, rect, size):
# 	"""Extracts a digit (if one exists) from a Sudoku square."""

# 	digit = cut_from_rect(img, rect)  # Get the digit box from the whole square

# 	# Use fill feature finding to get the largest feature in middle of the box
# 	# Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
# 	h, w = digit.shape[:2]
# 	margin = int(np.mean([h, w]) / 2.5)
# 	_, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
# 	digit = cut_from_rect(digit, bbox)

# 	# Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
# 	w = bbox[1][0] - bbox[0][0]
# 	h = bbox[1][1] - bbox[0][1]

# 	# Ignore any small bounding boxes
# 	if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
# 		return scale_and_centre(digit, size, 4)
# 	else:
# 		return np.zeros((size, size), np.uint8)


# def get_digits(img, squares, size):
#     """Extracts digits from their cells and builds an array"""
#     digits = []
#     img = pre_process_image(img.copy(), skip_dilate=True)
# #    cv2.imshow('img', img)
#     for square in squares:
#         digits.append(extract_digit(img, square, size))
#     return digits


# def parse_grid(path):
#     original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     processed = pre_process_image(original)
    
# #    cv2.namedWindow('processed',cv2.WINDOW_AUTOSIZE)
# #    processed_img = cv2.resize(processed, (500, 500))          # Resize image
# #    cv2.imshow('processed', processed_img)
    
#     corners = find_corners_of_largest_polygon(processed)
#     cropped = crop_and_warp(original, corners)
    
# #    cv2.namedWindow('cropped',cv2.WINDOW_AUTOSIZE)
# #    cropped_img = cv2.resize(cropped, (500, 500))              # Resize image
# #    cv2.imshow('cropped', cropped_img)
    
#     squares = infer_grid(cropped)
# #    print(squares)
#     digits = get_digits(cropped, squares, 28)
# #    print(digits)
#     final_image = show_digits(digits)
#     return final_image


# def extract_sudoku(image_path):
#     final_image = parse_grid(image_path)
#     return final_image

# def output(a):
#     sys.stdout.write(str(a))

# def display_sudoku(sudoku):
#     for i in range(9):
#         for j in range(9):
#             cell = sudoku[i][j]
#             if cell == 0 or isinstance(cell, set):
#                 output('.')
#             else:
#                 output(cell)
#             if (j + 1) % 3 == 0 and j < 8:
#                 output(' |')

#             if j != 8:
#                 output('  ')
#         output('\n')
#         if (i + 1) % 3 == 0 and i < 8:
#             output("--------+----------+---------\n")

# def extract_number(sudoku):
#     sudoku = cv2.resize(sudoku, (450,450))
# #    cv2.imshow('sudoku', sudoku)

#     # split sudoku
#     grid = np.zeros([9,9])
#     for i in range(9):
#         for j in range(9):
# #            image = sudoku[i*50+3:(i+1)*50-3,j*50+3:(j+1)*50-3]
#             image = sudoku[i*50:(i+1)*50,j*50:(j+1)*50]
# #            filename = "images/sudoku/file_%d_%d.jpg"%(i, j)
# #            cv2.imwrite(filename, image)
#             # if image.sum() > 25000:
#             #    # grid[i][j] = identify_number(image)
#             # else:
#             #     grid[i][j] = 0
#     return grid.astype(int)
    
# if __name__ == '__main__':
# 	image = extract_sudoku('sample.jpg')
# plt.imshow(image,'gray')
# plt.show()
# grid = extract_number(image)
# print('Sudoku:')
# display_sudoku(grid.tolist())



import numpy as np
import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
from mnist_loader import load_mnist_data, visualize_data, normalize_data
import digit_recognition
from keras.models import model_from_json

class ImageProcessing():
    # def __init__(self):
    #     print("a")

    def show_image(self, img):
        """Shows an image until any key is pressed"""
    #    print(type(img))
    #    print(img.shape)
    #    cv2.imshow('image', img)  # Display the image
    #    cv2.imwrite('images/gau_sudoku3.jpg', img)
    #    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    #    cv2.destroyAllWindows()  # Close all windows
        return img
    def show_digits(self, digits, colour=255):
        """Shows list of 81 extracted digits in a grid format"""
        rows = []
        with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
        for i in range(9):
            row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
            rows.append(row)
        img = self.show_image(np.concatenate(rows))
        return img

    def scale_and_centre(self, img, size, margin=0, background=0):
        """Scales and centres an image onto a new background square."""
        h, w = img.shape[:2]

        def centre_pad(length):
            """Handles centering for a given length that may be odd or even."""
            if length % 2 == 0:
                side1 = int((size - length) / 2)
                side2 = side1
            else:
                side1 = int((size - length) / 2)
                side2 = side1 + 1
            return side1, side2

        def scale(r, x):
            return int(r * x)

        if h > w:
            t_pad = int(margin / 2)
            b_pad = t_pad
            ratio = (size - margin) / h
            w, h = scale(ratio, w), scale(ratio, h)
            l_pad, r_pad = centre_pad(w)
        else:
            l_pad = int(margin / 2)
            r_pad = l_pad
            ratio = (size - margin) / w
            w, h = scale(ratio, w), scale(ratio, h)
            t_pad, b_pad = centre_pad(h)

        img = cv2.resize(img, (w, h))
        img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
        return cv2.resize(img, (size, size))

    def find_largest_feature(self, inp_img, scan_tl=None, scan_br=None):
        img = inp_img.copy()  # Copy the image, leaving the original untouched
        height, width = img.shape[:2]

        max_area = 0
        seed_point = (None, None)

        if scan_tl is None:
            scan_tl = [0, 0]

        if scan_br is None:
            scan_br = [width, height]

        # Loop through the image
        for x in range(scan_tl[0], scan_br[0]):
            for y in range(scan_tl[1], scan_br[1]):
                # Only operate on light or white squares
                if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
                    area = cv2.floodFill(img, None, (x, y), 64)
                    if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                        max_area = area[0]
                        seed_point = (x, y)

        # Colour everything grey (compensates for features outside of our middle scanning range
        for x in range(width):
            for y in range(height):
                if img.item(y, x) == 255 and x < width and y < height:
                    cv2.floodFill(img, None, (x, y), 64)

        mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

        # Highlight the main feature
        if all([p is not None for p in seed_point]):
            cv2.floodFill(img, mask, seed_point, 255)

        top, bottom, left, right = height, 0, width, 0

        for x in range(width):
            for y in range(height):
                if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                    cv2.floodFill(img, mask, (x, y), 0)

                # Find the bounding parameters
                if img.item(y, x) == 255:
                    top = y if y < top else top
                    bottom = y if y > bottom else bottom
                    left = x if x < left else left
                    right = x if x > right else right

        bbox = [[left, top], [right, bottom]]
        return img, np.array(bbox, dtype='float32'), seed_point


    def extract_digit(self, img, rect, size):
        """Extracts a digit (if one exists) from a Sudoku square."""

        digit = img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]  # Get the digit box from the whole square

        # Use fill feature finding to get the largest feature in middle of the box
        # Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
        h, w = digit.shape[:2]
        margin = int(np.mean([h, w]) / 2.5)
        _, bbox, seed = self.find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
        digit = digit[int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0])]

        # Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
        w = bbox[1][0] - bbox[0][0]
        h = bbox[1][1] - bbox[0][1]

        # Ignore any small bounding boxes
        if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
            return self.scale_and_centre(digit, size, 4)
        else:
            return np.zeros((size, size), np.uint8)

    def get_digits(self, img, squares, size):
        """Extracts digits from their cells and builds an array"""
        digits = []
        for square in squares:
            digits.append(self.extract_digit(img, square, size))
        return digits

    def get_squares(self, img, number_square_row):
        squares = []
        side = img.shape[:1]
        side = side[0]/number_square_row

        for i in range(number_square_row):
            for j in range(number_square_row):
                p1 = (i * side, j * side)  # Top left corner of a bounding box
                p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
                squares.append((p1, p2))
        return squares

    def get_cropped_image(self, img, corners):
        top_left = corners[0]
        top_right = corners[1]
        bottom_left = corners[2]
        bottom_right = corners[3]

        src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

        side = max([
        np.sqrt(((top_right[0] - bottom_right[0])**2) + ((top_right[1] - bottom_right[1])**2)),
		np.sqrt(((bottom_left[0] - top_left[0])**2) + ((bottom_left[1] - top_left[1])**2)),
        np.sqrt(((bottom_left[0] - bottom_right[0])**2) + ((bottom_left[1] - bottom_right[1])**2)),
		np.sqrt(((top_left[0] - top_right[0])**2) + ((top_left[1] - top_right[1])**2))
	    ])

        dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
        m = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, m, (int(side), int(side)))

    def get_corners(self, img):
        contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True) #Dscending Sort by area      
        polygon = contours[0]  # Largest image

        print(contours)
        # Bottom-right point has the largest (x + y) value
        # Top-left has point smallest (x + y) value
        # Bottom-left point has smallest (x - y) value
        # Top-right point has largest (x - y) value
        bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

        # Return an array of all 4 points using the indices
        # Each point is in its own array of one coordinate
        corners = [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]] 
        return corners

    def blurring(self, img, kernel_size):
        return cv2.GaussianBlur(img, kernel_size, 0)

    def adaptive_thresholding(self, img):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 11, 2)

    def invertion(self,img):
        return cv2.bitwise_not(img)

    def dilation(self, img, kernel):
        return cv2.dilate(img, kernel)

    def erosion(self, img, kernel):
        return cv2.erode(img, kernel, iterations=1) 

    def obtain_binary_image(self, img):
        kernel_size = (9,9)
        binary_image = self.blurring(img.copy(), kernel_size)
        binary_image = self.adaptive_thresholding(binary_image)
        binary_image = self.invertion(binary_image)
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
        binary_image = self.erosion(binary_image, kernel)
        return binary_image

    def process_sudoku(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = self.obtain_binary_image(image)
        #cv2.imshow('processed', image)
        plt.imshow(image,'gray')
        plt.title('Processed')
        plt.show()

        sudoku_corners = self.get_corners(image)
        plt.imshow(sudoku_corners,'gray')
        plt.title('Corners')
        plt.show()
        
        crop = self.get_cropped_image(image, sudoku_corners)
        plt.imshow(crop,'gray')
        
        plt.show()

        squares = self.get_squares(crop,9)
        digits = self.get_digits(crop, squares, 28)
        #print(digits)
        final_image = self.show_digits(digits)

        return image

    def identify_number(self, image, model):
        image_resize = cv2.resize(image, (28,28))    # For plt.imshow
        plt.imshow('image_resize', image_resize)
        plt.show()
        image_resize_2 = image_resize.reshape(1,28,28,1) 

        pred = model.predict(image_resize_2)
        return pred.argmax()

    def extract_number(self, sudoku, model):
        sudoku = cv2.resize(sudoku, (450,450))
        #plt.imshow('sudoku', sudoku)
        #plt.show()
        # split sudoku
        grid = np.zeros([9,9])
        for i in range(9):
            for j in range(9):
            #   image = sudoku[i*50+3:(i+1)*50-3,j*50+3:(j+1)*50-3]
                image = sudoku[i*50:(i+1)*50,j*50:(j+1)*50]
            #   filename = "images/sudoku/file_%d_%d.jpg"%(i, j)
            #   cv2.imwrite(filename, image)
                if image.sum() > 25000:
                    grid[i][j] = self.identify_number(image, model)
                else:
                    grid[i][j] = 0
        return grid.astype(int)

    def show_image(self, image):
        plt.imshow(image,'gray')
        plt.show()

if __name__ == '__main__':
    imageprocessing = ImageProcessing()
    image = imageprocessing.process_sudoku('sudoku.jpg')
    imageprocessing.show_image(image)

    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    x_train ,x_test = normalize_data(x_train, x_test)

    input_shape = (28, 28, 1)
    filters = 28
    kernel_size = (3, 3)
    pool_size = (2, 2)
    number_neurons_hidden = 128
    rate = 0.2
    number_neurons_output = 10
    epochs = 10

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # recognizer = digit_recognition.Recognizer(input_shape, filters, kernel_size, pool_size, number_neurons_hidden, rate, number_neurons_output, epochs)
    # model = recognizer.initialize_cnn()
    # model = recognizer.fit_model(model, x_train, y_train)
    # # print(recognizer.evaluate_model(model, x_test, y_test))
    grid = imageprocessing.extract_number(image, loaded_model)
    print('Sudoku:')
    print(grid.tolist())
