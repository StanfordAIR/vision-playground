import numpy as np
import cv2
import random
import math
from string import ascii_uppercase
import json
# NEED TO ADD THE REST OF THE CARDINAL DIRECTION TRANSFORMATIONS
NORTH = cv2.getRotationMatrix2D((300/2,300/2),0,1)
NORTH_EAST = cv2.getRotationMatrix2D((300/2,300/2),-45,1)
EAST = cv2.getRotationMatrix2D((300/2,300/2),-90,1)
SOUTH_EAST = cv2.getRotationMatrix2D((300/2,300/2),-135,1)
SOUTH = cv2.getRotationMatrix2D((300/2,300/2),-180,1)
SOUTH_WEST = cv2.getRotationMatrix2D((300/2,300/2),-225,1)
WEST = cv2.getRotationMatrix2D((300/2,300/2),-270,1);
NORTH_WEST = cv2.getRotationMatrix2D((300/2,300/2),-315,1)
cardinalDirections = [NORTH,NORTH_EAST,EAST, WEST,NORTH_WEST]
cardinalDirectionsStrings = ["NORTH", "NORTH_EAST", "EAST", "WEST", "NORTH_WEST"]
def drawRectangle(img):
    point1 = (80,80)
    point2 = (220,250)
    white_color = (255,255,255)
    # NOTE: Draws a rectangle
    cv2.rectangle(img, point1, point2, white_color, -1);
    return [point1[0], point1[1], point2[0]-point1[0], point2[1] - point1[1]];


def drawTrapezoid(img):
    trapezoid_center = [150, 150];
    b1 = 150;
    b2 = 250;
    white_color = (255,255,255)
    height = 100;
    point_1 = [trapezoid_center[0] - b1/2, trapezoid_center[1] - height/2]
    point_2 = [trapezoid_center[0] + b1/2, trapezoid_center[1] - height/2]
    point_3 = [trapezoid_center[0] - b2/2, trapezoid_center[1] + height/2]
    point_4 = [trapezoid_center[0] + b2/2, trapezoid_center[1] + height/2]
    pts = np.array([point_1, point_2, point_4, point_3], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img,[pts],white_color)
    return [point_3[1] - height, point_3[0], b2, height]

def drawPentagon(img):
    pentagon_center = [150, 150];
    axis = 100;
    points = [];
    white_color = (255,255,255)
    for i in range(0, 5):
        points.append([pentagon_center[0] + axis * math.cos(72 * i * 3.14/180), pentagon_center[1] + axis * math.sin(72 * i * 3.14/180)]);
    pts = np.array([points[0], points[1], points[2], points[3], points[4]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img,[pts], white_color)
    return [pentagon_center[0] - axis/2, pentagon_center[1]-axis/2, axis, axis]


def drawHexagon(img):
    hexagon_center = [150, 150];
    white_color = (255,255,255)
    axis = 100;
    points = [];
    for i in range(0, 6):
        points.append([hexagon_center[0] + axis * math.cos(60 * i * 3.14/180), hexagon_center[1] + axis * math.sin(60 * i * 3.14/180)]);
    pts = np.array([points[0], points[1], points[2], points[3], points[4], points[5]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img,[pts], white_color)
    return [hexagon_center[0] - axis/2, hexagon_center[1]-axis/2, axis, axis]

def drawHeptagon(img):
    heptagon_center = [150, 150];
    white_color = (255,255,255)
    axis = 100;
    points = [];
    for i in range(0, 7):
        points.append([heptagon_center[0] + axis * math.cos(51.4 * i * 3.14/180), heptagon_center[1] + axis * math.sin(51.4 * i * 3.14/180)]);
    pts = np.array([points[0], points[1], points[2], points[3], points[4], points[5], points[6]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img,[pts], white_color)
    return [heptagon_center[0] - axis/2, heptagon_center[1]-axis/2, axis, axis]

def drawOctagon(img):
    octagon_center = [150, 150];
    axis = 100;
    white_color = (255, 255, 255)
    points = [];
    for i in range(0, 8):
        points.append([octagon_center[0] + axis * math.cos(45 * i * 3.14/180), octagon_center[1] + axis * math.sin(45 * i * 3.14/180)]);
    pts = np.array([points[0], points[1], points[2], points[3], points[4], points[5], points[6], points[7]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img,[pts], white_color)
    return [octagon_center[0] - axis/2, octagon_center[1]-axis/2, axis, axis]

def drawStar(img):
    star_center = [150, 150];
    axis = 70;
    white_color = (255, 255, 255)
    points = [];
    for i in range(0, 5):
        points.append([star_center[0] + axis * math.cos((90 + 72 * -i) * 3.14/180), star_center[1] + axis * math.sin((90 + 72 * -i) * 3.14/180)]);
    points2 = [];
    axis = 150;
    for i in range(0, 5):
        points2.append([star_center[0] + axis * math.cos((-90 + 72 * -i) * 3.14/180), star_center[1] + axis * math.sin((-90 + 72 * -i) * 3.14/180)]);
    pts = np.array([points[0], points2[3], points[1], points2[4], points[2], points2[0], points[3], points2[1], points[4], points2[2]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img,[pts], white_color)
    return [star_center[0] - axis/2, star_center[1]-axis/2, axis, axis]

def drawCircle(img):
    circle_center = [150, 150]
    axis = 100;
    white_color = (255,255,255)
    cv2.circle(img,(circle_center[0], circle_center[1]), axis,white_color,-1)
    return [circle_center[0] - axis/2, circle_center[0] - axis/2, axis, axis];
# def drawSemiCircleConcaveUp(img):
#     white_color = (255,255,255)
#     center = (150,90)
#     axes = (125,125)
#     angle=0;
#     startAngle=0;
#     endAngle=180;
#     cv2.ellipse(img,center,axes,angle,startAngle,endAngle, white_color,-1);


def drawSemiCircleConcaveDown(img):
    center = (150,210)
    white_color = (255,255,255)
    axes = (125,125)
    angle = 0
    startAngle = 180
    endAngle = 360
    cv2.ellipse(img,center,axes,angle,startAngle,endAngle,white_color,-1)
    return (center[0] - axes[0], center[1] - axes[0], axes[1], axes[1]*2);

def padCharImg(data, boundary):
    padded_data = np.zeros((300,300,3),np.uint8)
    left_pos = [boundary[0]+boundary[3]/2 - data.shape[0]/2, boundary[1]+boundary[2]/2 - data.shape[1]/2];
    print left_pos
    padded_data[left_pos[0]:data.shape[0]+left_pos[0], left_pos[1]:(data.shape[1]+left_pos[1])] = data
    return padded_data

def pixelDistFromBlack(pixel1):
    BLACK_RGB = [0,0,0]
    r_diff = (pixel1[0]-BLACK_RGB[0])*(pixel1[0]-BLACK_RGB[0])
    g_diff = (pixel1[1]-BLACK_RGB[1])*(pixel1[1]-BLACK_RGB[1])
    b_diff = (pixel1[2]-BLACK_RGB[2])*(pixel1[2]-BLACK_RGB[2])
    return math.sqrt(r_diff+g_diff+b_diff)

# Function is passed a blank white shape img + a black-on-white alphanum img
# NOTE: this function can pass an img background with black color (which wouldn't contrast with the already black background
#       surrounding the shape)
def colorShapeIMG(img,new_dst):
    WHITE_RGB = [255,255,255]
    # White, Black, Gray, Red, Blue, Green, Yellow, Purple, Brown, Orange
    colors = [(255,255,255),(15,15,15),(128,128,128),(30,30,240),
              (230,30,200),(30,240,30),(20,230,230),(128,0,128),(59,92,131),(0,165,255)]
    color_string = ["White", "Black", "Gray", "Red", "Blue", "Green", "Yellow", "Purple", "Brown", "Orange"]
    alpha_numeric_choice = int(random.random()*10)
    shape_choice = int(random.random() * 10);
    random_background_color = colors[shape_choice]
    random_alphanum_color = colors[alpha_numeric_choice]
    # Makes sure the alpanum color and the background color are different
    choice = 0;
    while(random_background_color == random_alphanum_color):
        alpha_numeric_choice = int(random.random()*10)
        random_alphanum_color = colors[alpha_numeric_choice]

    for i in range(0,len(img)):
        for x in range(0,len(img[i])):
            if np.all(img[i][x]==WHITE_RGB):
                if pixelDistFromBlack(new_dst[i][x]) < 200:
                    img[i][x] = random_alphanum_color
                else:
                    img[i][x] = random_background_color
    return img, color_string[alpha_numeric_choice], color_string[shape_choice]


def overlayChar(data, img, boundary):
    WHITE_RGB = [255,255,255]
    # Inverts the color

    charNum = int(len(data)*random.random())
    # NOTE: OpenCV resize transformation (handwritten character is slighly smaller than the background img)
    new_dst = padCharImg(cv2.resize(data[charNum], (data[charNum].shape[1]*2, data[charNum].shape[0]*2), interpolation=cv2.INTER_NEAREST), boundary);

    # PREVIOUSLY: This is subtracted in order to convert the new_dst white (255,255,255) rgb values to black values (0,0,0)
    # return img-new_dst

    new_dst = 255-new_dst
    # Returns a random background / alphanumeric combination
    img, lettercolor, shapecolor = colorShapeIMG(img,new_dst)
    return img, charNum, lettercolor, shapecolor

def genImage(overlay_img):
    big_img = cv2.imread("background.png");
    # Randomly selected top left corner of the new generated image
    top_left = [int(random.random() * 100), int(random.random() * 100)];
    # the bottom right corner
    bottom_right = [top_left[0] + 300, top_left[1] + 300];
    # new random image square is cropped out
    cropped_img = big_img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]];
    M = np.float32([[1,0,int(random.random() * 100)-50],[0,1,int(random.random() * 100)-50]])
    overlay_img = cv2.warpAffine(overlay_img,M,(300,300))

    roi = cropped_img[0:300, 0:300]
    #create mask
    img2gray = cv2.cvtColor(overlay_img,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
    img2_fg = cv2.bitwise_and(overlay_img,overlay_img,mask = mask)
    dst = cv2.add(img1_bg,img2_fg)
    cropped_img[0:300, 0:300] = dst
    return cropped_img;

def readDataset():
    data = [];
    for c in ascii_uppercase:
        img = cv2.imread('Chars/'+c+'Text.png');
        img = cv2.resize(img,(46,65),interpolation=cv2.INTER_NEAREST)
        y_crop_loc = 0
        for i in range(23, 46):
            flag = True;
            for j in range(0, 65):
                if (img[j][i][0] != 255):
                    flag = False;
            if (flag==True):
                y_crop_loc = i + 5;
                break;
        x_crop_loc = 0
        for i in range(25, 65):
            flag = True;
            for j in range(1, 46):
                if (img[i][j][0] != 255):
                    flag = False;
            if (flag==True):
                x_crop_loc = i + 5;
                break;
        if(x_crop_loc == 0):
            x_crop_loc = 65;
        img = img[0:x_crop_loc, 0:y_crop_loc];
        data.append(255-img);
    return data

def drawImageAndTransform(img):
    metadata = dict()
    draw_shape_string = ["Hexagon", "Circle", "Star", "Octagon", "Heptagon",
                         "Pentagon", "Trapezoid", "Rectangle", "SemiCircle"]
    shape_num = int(random.random() * 9);
    metadata["Shape"] = draw_shape_string[shape_num];
    boundary = draw_functions[shape_num](img);
    # Overlays the "text" image over the image
    dst, which_char, letter_color, shape_color = overlayChar(data, img, boundary);
    metadata["Char"] = str(chr(which_char+65))
    print str(chr(which_char+65))
    metadata["ShapeColor"] = shape_color;
    metadata["LetterColor"] = letter_color
    # APPLIES A CARDINAL ORIENTATION TRANSFORMATION
    direction_num = int(random.random()*5)
    dst = cv2.warpAffine(img, cardinalDirections[direction_num], (300,300))
    metadata["Orientation"] = cardinalDirectionsStrings[direction_num];
    # Overlays the shape + character image on the background image
    dst = genImage(dst);
    
    return dst, metadata

# Creates the alphanumeric "text" image array
data = readDataset();
draw_functions = [drawHexagon, drawCircle, drawStar, drawOctagon, drawHeptagon,
                  drawPentagon, drawTrapezoid, drawRectangle, drawSemiCircleConcaveDown]
labels = dict();
for i in range(10000):
    #  Creates a black image array (height,width,rgb)
    img = np.zeros((300, 300, 3), np.uint8);
    # Draws a white shape on the black image
    img, metadata = drawImageAndTransform(img);
    img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_NEAREST);
    # APPLIES a gaussian blur effect
    gaussian_blur = cv2.GaussianBlur(img,(15,15),0);
    labels[str(i) + '.png'] = metadata;
    cv2.imwrite('dataset/' + str(i) + '.png', img)

with open('labels.json', 'w') as fp:
    json.dump(labels, fp)
