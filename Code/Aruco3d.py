import cv2
import cv2.aruco as aruco
import aruco_module as aru
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from Calibracion import *
from object_module import *
from objloader_simple import *
import math

DEFAULT_COLOR = (0, 0, 0)

def main():
    cap = cv2.VideoCapture(1) # Selecciona la entrada de video, 0 para la camara del pc y 1 para cualquier camara externa
    
    ## obj = three_d_object('obj/charmander.obj', None)
    obj = three_d_object('obj/fox3.obj', 'obj/texture.png')
    ## obj = three_d_object('obj/torytur.obj', 'obj/metallic.jpg')
    ## obj = three_d_object('obj/beldum.obj', None)

    #dir_name = os.getcwd()
    #obj = OBJ(os.path.join(dir_name, 'obj/charmander.obj'), swapyz=True)
    
    calib= calibracion()    # Obejto calibracion 
    CameraMatrix, dist, esquinas = calib.calibracion_cam() # Devuelve la matriz de calibración de la camara de mi móvil y los valores de distorsión
    
    marker = EncuentraAruco("Markers/2.png")  # Devuelve la imagen redimensionada del marcador que he elegido
    sigs = sigCalculation(marker)
    while True:
        sccuuess, frame = cap.read() 
        arucoFound = findArucoMarkers(frame) # devuelve [bbox,id]
        ## success, Homografia = aru.find_homography_aruco(frame, marker, sigs)
        
        if len(arucoFound[0])!=0:
            rvecs, tvecs, objPoints = calculo3daruco(frame, arucoFound[0],CameraMatrix,dist, esquinas)  
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                None
                frame = augmentAruco(frame,bbox, id, frame, marker, CameraMatrix,obj)     # ---------------- QUITAR 2--------------
            
            dibujarCubo(frame,arucoFound[0])  #Dibuja algo parecido a un cubo xd       --------------------QUITAR 1----------------
        
        ## if not success:
			# print('homograpy est failed')
            ## cv2.imshow("Image", frame)
            ## cv2.waitKey(1)
            ## continue
        
       
        cv2.imshow("Image", frame)
        cv2.waitKey(1)
        
def findArucoMarkers(img, markerSize = 6, totalMarkers = 250, draw = True):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgGray = binarize_kmeans(gaussian_smoothing(imgGray,0.2,1),5)
    ## cv2.imshow("gray", imgGray)  ## Muestra la imagen binarizada
    key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = cv2.aruco.detectMarkers(imgGray,arucoDict,parameters = arucoParam) 
    
    
    if draw:
        aruco.drawDetectedMarkers(img,bboxs)
        
        
    return [bboxs,ids]

def calculo3daruco(img, bbox, cameraMatrix, distorsion,esquinas, draw = True):
    
    rvecs, tvecs, objPoints = aruco.estimatePoseSingleMarkers(bbox,0.65,cameraMatrix,distorsion)
    try:    
        if draw:
            ## rvecs2, tvecs2 ,_= cv2.solvePnP(objPoints,esquinas,cameraMatrix,distorsion,)
            (rvecs - tvecs).any()
            cv2.drawFrameAxes(img,cameraMatrix,distorsion,rvecs,tvecs,0.1)
    except:
        None
        
    return rvecs, tvecs, objPoints
 
def dibujarCubo(frame,bbox):
    tl = (bbox[0][0][0][0], bbox[0][0][0][1])
    tr = (bbox[0][0][1][0], bbox[0][0][1][1])
    br = (bbox[0][0][2][0], bbox[0][0][2][1])
    bl = (bbox[0][0][3][0], bbox[0][0][3][1])
    
    v1, v2 = tl[0],tl[1]
    v3, v4 = tr[0],tr[1]
    v5, v6 = br[0],br[1]
    v7, v8 = bl[0],bl[1]
    
    # Cara Inferior
    cv2.line(frame, (int(v1), int(v2)), (int(v3), int(v4)), (255,255,0),3)
    cv2.line(frame, (int(v5), int(v6)), (int(v7), int(v8)), (255,255,0),3)
    cv2.line(frame, (int(v1), int(v2)), (int(v7), int(v8)), (255,255,0),3)
    cv2.line(frame, (int(v3), int(v4)), (int(v5), int(v6)), (255,255,0),3)
    
    #Cara superior 
    cv2.line(frame, (int(v1), int(v2-200)), (int(v3), int(v4-200)), (255,255,0),3)
    cv2.line(frame, (int(v5), int(v6-200)), (int(v7), int(v8-200)), (255,255,0),3)
    cv2.line(frame, (int(v1), int(v2-200)), (int(v7), int(v8-200)), (255,255,0),3)
    cv2.line(frame, (int(v3), int(v4-200)), (int(v5), int(v6-200)), (255,255,0),3)
    
    #Caras laterales
    cv2.line(frame, (int(v1), int(v2-200)), (int(v1), int(v2)), (255,255,0),3)
    cv2.line(frame, (int(v3), int(v4-200)), (int(v3), int(v4)), (255,255,0),3)
    cv2.line(frame, (int(v5), int(v6-200)), (int(v5), int(v6)), (255,255,0),3)
    cv2.line(frame, (int(v7), int(v8-200)), (int(v7), int(v8)), (255,255,0),3)
    
def sigCalculation(marker):
    h,w = marker.shape
	#considering all 4 rotations
    marker_sig1 = aru.get_bit_sig(marker, np.array([[0,0],[0,w], [h,w], [h,0]]).reshape(4,1,2))
    marker_sig2 = aru.get_bit_sig(marker, np.array([[0,w], [h,w], [h,0], [0,0]]).reshape(4,1,2))
    marker_sig3 = aru.get_bit_sig(marker, np.array([[h,w],[h,0], [0,0], [0,w]]).reshape(4,1,2))
    marker_sig4 = aru.get_bit_sig(marker, np.array([[h,0],[0,0], [0,w], [h,w]]).reshape(4,1,2))

    sigs = [marker_sig1, marker_sig2, marker_sig3, marker_sig4]
    
    return sigs
   
def augmentAruco(frame,bbox, id, img,marker,CameraMatrix, obj):   # La tienes de referencia, quitala luego si no te sirve
    
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]
    
    h, w= marker.shape
    
    pts1 = np.array([tl,tr,br,bl])
    pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    Homografia, _ = cv2.findHomography(pts2,pts1)
    
    R_T = aru.get_extended_RT(CameraMatrix, Homografia)
    transformation = CameraMatrix.dot(R_T) 
        
    augmented = augment(frame, obj, transformation, marker)
    
    ##projection = projection_matrix(CameraMatrix,Homografia)
    ##augmented = render(frame,obj,projection,marker,False)

    return augmented

def gaussian_smoothing(image, sigma, w_kernel):
    """ Blur and normalize input image.   
    
        Args:
            image: Input image to be binarized
            sigma: Standard deviation of the Gaussian distribution
            w_kernel: Kernel aperture size
                    
        Returns: 
            smoothed_norm: Blurred image
    """   
    # Write your code here!
    
    # Define 1D kernel
    s=sigma
    w=w_kernel
    kernel_1D = np.array([(1/(s*np.sqrt(2*np.pi)))*np.exp(-((pow(z,2))/(2*pow(s,2)))) for z in range(-w,w+1)])
    
    # Apply distributive property of convolution
    vertical_kernel = kernel_1D.reshape(2*w+1,1)
    horizontal_kernel = kernel_1D.reshape(1,2*w+1)   
    gaussian_kernel_2D = signal.convolve2d(vertical_kernel, horizontal_kernel)   
    
    # Blur image
    smoothed_img = cv2.filter2D(image,cv2.CV_16S,gaussian_kernel_2D)
    
    # Normalize to [0 254] values
    smoothed_norm = np.array(image.shape)
    smoothed_norm = cv2.normalize(smoothed_img ,smoothed_norm , 0, 255, cv2.NORM_MINMAX) # Leave the second argument as None
    
    return smoothed_norm

def binarize_kmeans(image,it):
    """ Binarize an image using k-means.   

        Args:
            image: Input image
            it: K-means iteration
    """    
    
    # Set random seed for centroids 
    cv2.setRNGSeed(124)
    
    # Flatten image
    flattened_img = image.reshape((-1,1))
    flattened_img = np.float32(flattened_img)
    
    #Set epsilon
    epsilon = 0.2
    
    # Estabish stopping criteria (either `it` iterations or moving less than `epsilon`)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, it, epsilon)
    
    # Set K parameter (2 for thresholding)
    K = 2
    
    # Call kmeans using random initial position for centroids
    _,label,center=cv2.kmeans(flattened_img,K,None,criteria,it,cv2.KMEANS_RANDOM_CENTERS)
    
    # Colour resultant labels
    center = np.uint8(center) # Get center coordinates as unsigned integers   
    ##print(center)
    flattened_img = center[label.flatten()] # Get the color (center) assigned to each pixel
    
    # Reshape vector image to original shape
    binarized = flattened_img.reshape((image.shape))
    
    return binarized

def EncuentraAruco(path):
    marker_colored = cv2.imread(path)
    assert marker_colored is not None, "Could not find the aruco marker image file"

    marker_colored =  cv2.resize(marker_colored, (480,480), interpolation = cv2.INTER_CUBIC )
    marker = cv2.cvtColor(marker_colored, cv2.COLOR_BGR2GRAY)
    
    return marker

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))



if __name__ == '__main__':
    main()