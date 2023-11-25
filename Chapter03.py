import numpy as np
import cv2


import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

L = 256
#Converts an image to its negative by subtracting pixel values from the maximum intensity value (255 for an 8-bit image).
def Negative(imgin):

    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    for x in range(M):
        for y in range(N):
            r = imgin[x, y]
            s = L - 1 - r
            imgout[x, y] = s
    return imgout

#: Enhances low-intensity pixels using a logarithmic scale.
def Logarit(imgin):
    M, N,_ = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    c = (L-1)/np.log(L)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            if r == 0:
                r = 1
            s = c*np.log(1+r)
            imgout[x,y] = np.uint8(s)
    return imgout

#Adjusts the pixel values using a power-law curve, controlled by the parameter gamma.
def Power(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    gamma = 5.0
    c = np.power(L-1,1-gamma)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            s = c*np.power(r,gamma)
            imgout[x,y] = np.uint8(s)
    return imgout

#Performs a linear mapping of pixel intensities within specific ranges. This can enhance contrast in specific intensity ranges.
def PiecewiseLinear(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    rmin, rmax, vi_tri_rmin, vi_tri_rmax = cv2.minMaxLoc(imgin)
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = L-1
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            if r < r1:
                s = s1/r1*r
            elif r < r2:
                s = (s2-s1)/(r2-r1)*(r-r1) + s1
            else:
                s = (L-1-s2)/(L-1-r2)*(r-r2) + s2
            imgout[x,y] = np.uint8(s)
    return imgout

#Generates a histogram of the input image, showing the distribution of pixel intensities.
def Histogram(imgin):
    M, N,_ = imgin.shape
    imgout = np.zeros((M,L), np.uint8) + 255
    h = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            h[r] = h[r]+1
    p = h/(M*N)
    scale = 2000
    for r in range(0, L):
        cv2.line(imgout,(r,M-1),(r,M-1-int(scale*p[r])), (0,0,0))
    return imgout

#: Enhances the contrast of the image by spreading out the most frequent intensity values.
def HistEqual(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    h = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            h[r] = h[r]+1
    p = h/(M*N)

    s = np.zeros(L, np.float64)
    for k in range(0, L):
        for j in range(0, k+1):
            s[k] = s[k] + p[j]

    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            imgout[x,y] = np.uint8((L-1)*s[r])
    return imgout

#Applies histogram equalization separately to each color channel (Blue, Green, Red) of a color image.
def HistEqualColor(imgin):
    B = imgin[:,:,0]
    G = imgin[:,:,1]
    R = imgin[:,:,2]
    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)
    imgout = np.array([B, G, R])
    imgout = np.transpose(imgout, axes = [1,2,0]) 
    return imgout

# Applies histogram equalization within a local neighborhood of each pixel, useful for improving local contrast.
def LocalHist(imgin):
    M, N,_ = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    m = 3
    n = 3
    w = np.zeros((m,n), np.uint8)
    a = m // 2
    b = n // 2
    for x in range(a, M-a):
        for y in range(b, N-b):
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    w[s+a,t+b] = imgin[x+s,y+t]
            w = cv2.equalizeHist(w)
            imgout[x,y] = w[a,b]
    return imgout

#Adjusts pixel values based on local and global mean and standard deviation, useful for adaptive contrast enhancement.
def HistStat(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    m = 3
    n = 3
    w = np.zeros((m,n), np.uint8)
    a = m // 2
    b = n // 2
    mG, sigmaG = cv2.meanStdDev(imgin)
    C = 22.8
    k0 = 0.0
    k1 = 0.1
    k2 = 0.0
    k3 = 0.1
    for x in range(a, M-a):
        for y in range(b, N-b):
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    w[s+a,t+b] = imgin[x+s,y+t]
            msxy, sigmasxy = cv2.meanStdDev(w)
            r = imgin[x,y]
            if (k0*mG <= msxy <= k1*mG) and (k2*sigmaG <= sigmasxy <= k3*sigmaG):
                imgout[x,y] = np.uint8(C*r)
            else:
                imgout[x,y] = r
    return imgout

#Applies a box filter (averaging filter) with a custom kernel size to smooth the image.
def MyBoxFilter(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    m = 11
    n = 11
    w = np.ones((m,n))
    w = w/(m*n)

    a = m // 2
    b = n // 2
    for x in range(a, M-a):
        for y in range(b, M-b):
            r = 0.0
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    r = r + w[s+a,t+b]*imgin[x+s,y+t]
            imgout[x,y] = np.uint8(r)
    return imgout

#: Smoothes the image using OpenCV's built-in box filter function.
def BoxFilter(imgin):
    m = 21
    n = 21
    w = np.ones((m,n))
    w = w/(m*n)
    imgout = cv2.filter2D(imgin,cv2.CV_8UC1,w)
    return imgout
#Applies a fixed-level threshold to the image after blurring, useful for segmenting objects from the background.
def Threshold(imgin):
    temp = cv2.blur(imgin, (15,15))
    retval, imgout = cv2.threshold(temp,64,255,cv2.THRESH_BINARY)
    return imgout
# Reduces noise by replacing each pixel's value with the median value of the pixels in its neighborhood.
def MedianFilter(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    m = 5
    n = 5
    w = np.zeros((m,n), np.uint8)
    a = m // 2
    b = n // 2
    for x in range(0, M):
        for y in range(0, N):
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    w[s+a,t+b] = imgin[(x+s)%M,(y+t)%N]
            w_1D = np.reshape(w, (m*n,))
            w_1D = np.sort(w_1D)
            imgout[x,y] = w_1D[m*n//2]
    return imgout

def Sharpen(imgin):
    # Đạo hàm cấp 2 của ảnh
    w = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    temp = cv2.filter2D(imgin,cv2.CV_32FC1,w)

    # Hàm cv2.Laplacian chỉ tính đạo hàm cấp 2
    # cho bộ lọc có số -4 chính giữa
    imgout = imgin - temp
    imgout = np.clip(imgout, 0, L-1)
    imgout = imgout.astype(np.uint8)
    return imgout
 
def Gradient(imgin):
    sobel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    # Đạo hàm cấp 1 theo hướng x
    mygx = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_x)
    # Đạo hàm cấp 1 theo hướng y
    mygy = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_y)

    # Lưu ý: cv2.Sobel có hướng x nằm ngang
    # ngược lại với sách Digital Image Processing
    gx = cv2.Sobel(imgin,cv2.CV_32FC1, dx = 1, dy = 0)
    gy = cv2.Sobel(imgin,cv2.CV_32FC1, dx = 0, dy = 1)

    imgout = abs(gx) + abs(gy)
    imgout = np.clip(imgout, 0, L-1)
    imgout = imgout.astype(np.uint8)
    return imgout
#######################################

def save_image():
    global adjusted
    filepath = filedialog.asksaveasfilename(defaultextension=".jpg")
    if filepath:
        cv2.imwrite(filepath, adjusted)  # Lưu ảnh đã chỉnh sửa
        messagebox.showinfo("Thông báo", "Ảnh đã được lưu thành công!")

def on_trackbar_change(_):
   
    #adjusted = adjust_brightness(adjusted, brightness_alpha, brightness_beta)   
    global adjusted,image
    
    # Check the position of each trackbar and apply the corresponding transformation
    if cv2.getTrackbarPos('Negative', 'Trackbars') == 1:
        adjusted = Negative(image)

    if cv2.getTrackbarPos('Logarit', 'Trackbars') == 1:
        adjusted = Logarit(adjusted)

    if cv2.getTrackbarPos('Power', 'Trackbars') == 1:
        adjusted = Power(adjusted)

    if cv2.getTrackbarPos('PiecewiseLinear', 'Trackbars') == 1:
        adjusted = PiecewiseLinear(adjusted)

    if cv2.getTrackbarPos('Histogram', 'Trackbars') == 1:
        adjusted = Histogram(adjusted)

    if cv2.getTrackbarPos('HistEqual', 'Trackbars') == 1:
        adjusted = HistEqual(adjusted)

    if cv2.getTrackbarPos('HistEqualColor', 'Trackbars') == 1:
        adjusted = HistEqualColor(adjusted)

    if cv2.getTrackbarPos('LocalHist', 'Trackbars') == 1:
        adjusted = LocalHist(adjusted)

    if cv2.getTrackbarPos('HistStat', 'Trackbars') == 1:
        adjusted = HistStat(adjusted)

    if cv2.getTrackbarPos('MyBoxFilter', 'Trackbars') == 1:
        adjusted = MyBoxFilter(adjusted)

    if cv2.getTrackbarPos('BoxFilter', 'Trackbars') == 1:
        adjusted = BoxFilter(adjusted)

    if cv2.getTrackbarPos('Threshold', 'Trackbars') == 1:
        adjusted = Threshold(adjusted)

    if cv2.getTrackbarPos('MedianFilter', 'Trackbars') == 1:
        adjusted = MedianFilter(adjusted)

    if cv2.getTrackbarPos('Sharpen', 'Trackbars') == 1:
        adjusted = Sharpen(adjusted)

    if cv2.getTrackbarPos('Gradient', 'Trackbars') == 1:
        adjusted = Gradient(adjusted)

    
    #combined_img = np.hstack((image,adjusted))
    #cv2.imshow('Image Processing', combined_img)
    cv2.imshow('Image', adjusted)

def open_image_editor():
    global image, adjusted , initial_solarization
    filepath = filedialog.askopenfilename()
    if filepath:
        image = cv2.imread(filepath)
        # Convert to grayscale if the input image is color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        adjusted = image.copy()
        
           # Tạo cửa sổ mới cho các thanh trượt
        trackbar_window = "Trackbars"
        cv2.namedWindow(trackbar_window)
        cv2.resizeWindow(trackbar_window, 1600, 600)  # Đặt kích thước cho cửa sổ
          # Thêm các thanh trượt vào cửa sổ này
        
        # Create trackbars for color change and gamma adjustment
        
        cv2.createTrackbar('Negative', trackbar_window, 0,1, on_trackbar_change)
        cv2.createTrackbar('Logarit', trackbar_window, 0,1, on_trackbar_change)
        cv2.createTrackbar('Power', trackbar_window, 0,1, on_trackbar_change)
        cv2.createTrackbar('PiecewiseLinear', trackbar_window, 0,1, on_trackbar_change)
        cv2.createTrackbar('Histogram', trackbar_window, 0,1, on_trackbar_change)
        cv2.createTrackbar('HistEqual', trackbar_window, 0,1, on_trackbar_change)
        cv2.createTrackbar('HistEqualColor', trackbar_window, 0,1, on_trackbar_change)
        cv2.createTrackbar('LocalHist', trackbar_window, 0,1, on_trackbar_change)
        cv2.createTrackbar('HistStat', trackbar_window, 0,1, on_trackbar_change)
        cv2.createTrackbar('MyBoxFilter', trackbar_window, 0,1, on_trackbar_change)
        cv2.createTrackbar('BoxFilter', trackbar_window, 0,1, on_trackbar_change)
        cv2.createTrackbar('Threshold', trackbar_window, 0,1, on_trackbar_change)
        cv2.createTrackbar('MedianFilter', trackbar_window, 0,1, on_trackbar_change)
        cv2.createTrackbar('Sharpen', trackbar_window, 0,1, on_trackbar_change)
        cv2.createTrackbar('Gradient', trackbar_window, 0,1, on_trackbar_change)
        # Show the image
        cv2.namedWindow("Image")
        on_trackbar_change(0)  # Initial call to display the 
        cv2.waitKey(0) # Wait for a key press and then terminate the program
        cv2.destroyAllWindows()


image = None
adjusted = None
# Create a window
root = tk.Tk()
root.geometry("400x600")
# Calculate the position for the center of the screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - 400) / 2
y = (screen_height - 600) / 2

# Set the root window position to the center
root.geometry("+%d+%d" % (x, y))

#create open file dialog
open_button = tk.Button(root, text="Open image file", command=open_image_editor)
open_button.pack()

save_button = tk.Button(root, text="Save adjusted image",command=save_image)
save_button.pack()
root.mainloop()














