import cv2
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from os.path import basename, splitext
from skimage.measure import compare_ssim
from skimage.metrics import mean_squared_error
from skimage import img_as_float
import math


def get_sparse_neighbor(p, n, m):

    i, j = p // m, p % m
    d = {}
    if i - 1 >= 0:
        d[(i - 1) * m + j] = (i - 1, j, 0)
    if i + 1 < n:
        d[(i + 1) * m + j] = (i + 1, j, 0)
    if j - 1 >= 0:
        d[i * m + j - 1] = (i, j - 1, 1)
    if j + 1 < m:
        d[i * m + j + 1] = (i, j + 1, 1)

    return d



def gaussianFilter(size,sig):
    s=int(size/2) #to center it
    g=np.zeros((size,size),np.float64) #kernel type float
        
    for x in range(-s,s-1):
        for y in range(-s,s-1):
            g[x+s,y+s]=(1/(2*np.pi*(sig**2)))*np.exp(-(((x**2)+(y**2))/(2*(sig**2)))) #gaussian formula
    return g


def compute_smoothness_weights(L, x, kernel, eps):

    Lp = cv2.Sobel(L, cv2.CV_64F, int(x == 1), int(x == 0), ksize=1)
    T = convolve(np.ones_like(L), kernel, mode='constant')
    T = T / (np.abs(convolve(Lp, kernel, mode='constant')) + eps)
    return T / (np.abs(Lp) + eps)


def refine_illumination_map_linear(L, gamma, lamb, kernel, eps):

    # compute smoothness weights
    wx = compute_smoothness_weights(L, x=1, kernel=kernel, eps=eps)
    wy = compute_smoothness_weights(L, x=0, kernel=kernel, eps=eps)

    n, m = L.shape
    L_1d = L.copy().flatten()

    # compute the five-point spatially inhomogeneous Laplacian matrix
    row, column, data = [], [], []
    for p in range(n * m):
        diag = 0
        for q, (k, l, x) in get_sparse_neighbor(p, n, m).items():
            weight = wx[k, l] if x else wy[k, l]
            row.append(p)
            column.append(q)
            data.append(-weight)
            diag += weight
        row.append(p)
        column.append(p)
        data.append(diag)
    F = csr_matrix((data, (row, column)), shape=(n * m, n * m))

    # solve the linear system
    Id = diags([np.ones(n * m)], [0])
    A = Id + lamb * F
    L_refined = spsolve(csr_matrix(A), L_1d, permc_spec=None, use_umfpack=True).reshape((n, m))

    # gamma correction
    L_refined = np.clip(L_refined, a_min=eps, a_max=1) ** gamma

    return L_refined


def correct_underexposure(im, gamma, lamb, kernel, eps):

    # first estimation of the illumination map
    L = np.max(im, axis=-1)
    # illumination refinement
    L_refined = refine_illumination_map_linear(L, gamma, lamb, kernel, eps)

    # correct image underexposure
    L_refined_3d = np.repeat(L_refined[..., None], 3, axis=-1)
    im_corrected = im / L_refined_3d #divide original image by this refined illumination map
    return im_corrected


def enhance_image_exposure(img, gamma, lamb, sigma, eps):

    print(img)
    lowLightImage=cv2.imread(img)
    dim=lowLightImage.shape
    
    if(dim[0]>3000 or dim[1]>3000):
        lowLightImage=cv2.resize(lowLightImage, (int(dim[1]/3.0),int(dim[0]/3.0)), interpolation = cv2.INTER_AREA) #resize to smaller size for efficienc
    elif(dim[0]>1000 or dim[1]>1000):
        lowLightImage=cv2.resize(lowLightImage, (int(dim[1]/2.0),int(dim[0]/2.0)), interpolation = cv2.INTER_AREA) #resize to smaller size for efficiency

    # create spacial affinity kernel/ gaussian kernel
    radius=np.floor(4.0*sigma+0.5)
    size = 2*radius + 1
    size=int(size)
    kernel=gaussianFilter(size, sigma)#get gaussian kernel

    # correct underexposudness
    imgNorm = lowLightImage.astype(float) / 255.0 #normalize
    imgLowLightEnhanced = correct_underexposure(imgNorm, gamma, lamb, kernel, eps)


    # convert to 8 bits and returns
    return np.clip(imgLowLightEnhanced*255, a_min=0, a_max=255).astype("uint8")

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def lowLightComparison(numImg):
    
    mseTotal=0
    #ssimTotal=0
    psnrTotal=0
 
    for i in range(1,numImg):
        pathEnhanced="dark/lowLightEnhance/low ("+str(i)+")Enhanced.png"
        pathTrue="true/normal ("+str(i)+").png"
      
        enhancedImg=cv2.imread(pathEnhanced)
        enhancedImgGray=cv2.cvtColor(enhancedImg.copy(), cv2.COLOR_BGR2GRAY)
        enhancedImg=img_as_float(enhancedImg) 
        
        trueImg=cv2.imread(pathTrue)
        trueImgGray=cv2.cvtColor(trueImg.copy(), cv2.COLOR_BGR2GRAY)
        trueImg=img_as_float(trueImg)
        
        
        #ssimVal,_= compare_ssim(trueImgGray, enhancedImgGray, full=True)     
        #ssimTotal=ssimTotal+ssimVal
        
        mse=mean_squared_error(trueImg, enhancedImg)
        mseTotal=mseTotal+mse
        
        psnr=20 * math.log10(255.0/math.sqrt(mse))
        psnrTotal=psnrTotal+psnr
        
    mseAvg=mseTotal/(numImg-1)
    #ssimAvg=ssimTotal/(numImg-1)
    psnrAvg=psnrTotal/(numImg-1)
    
    print("Average Mean Squared Error between real image and low light enhanced image:",mseAvg,"\n")  
    #print("Average SSIM between real image and low light enhanced image:",ssimAvg,"\n")
    print("Average PSNR between real image and low light enhanced image:",psnrAvg,"\n")          
      


#Low light enhance
def limeLowLightImprove(folder,folder2,imgName,numImg,saveLoc):
    gamma=0.61 #0.6
    lamb=0.16 #0.15
    sigma=5  #3
    eps=1e-3 #1e-3
    
    # load images
    imdir = folder
    images = []
    for i in range(1,numImg):
        file="/"+folder2+"/"+imgName+" ("+str(i)+").png"
    
        images.append(imdir+file)
    
    # create save directory
    enhDir = imdir+"/"+saveLoc
    
    
    i=0
    for img in images :
        
        lowLightEnhImage = enhance_image_exposure(img, gamma, lamb, sigma, eps)

        filename = basename(images[i])
        name, ext = splitext(filename)
    
        enhancedImage = name+"Enhanced"+ext
        enhancedPath=enhDir+"/"+enhancedImage
    
        cv2.imwrite(enhancedPath, lowLightEnhImage)
        i=i+1
    print("Low light image enhancement completed\n")
    

#Face Detect

def faceDetect(folder,numImg,name,saveLoc):

    tot=0
    for i in range(1,numImg):
        path="DarkFace/"+folder+"/"+"darkFace ("+str(i)+")"+name+".png"
        
        # Read image
        origImg = cv2.imread(path)
    
        # Convert color image to grayscale for Viola-Jones
        grayImg = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)
    
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    
        detected_faces = face_cascade.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=1)
    
        for (column, row, width, height) in detected_faces:
            cv2.rectangle(origImg,(column, row),(column + width, row + height),(0, 255, 0),3) #draws green rectangle
    
        faceDetectImg=convertToRGB(origImg)
        
        faceDetectPath="DarkFace/"+saveLoc+"/faceDetectImg"+str(i)+".png"
        cv2.imwrite(faceDetectPath, convertToRGB(faceDetectImg))
        tot=tot+len(detected_faces)
    
    print("Total faces detected:",tot)
    print("Done Face Detect\n")
    
    
#run code    
limeLowLightImprove('dark','LowLight','low',790,'lowLightEnhance')#790
limeLowLightImprove('DarkFace','lowFace','darkFace',301,'FaceLowLightEnhance')#301

faceDetect('lowFace',301,'','FaceDetectLow')#301
faceDetect('FaceLowLightEnhance',301,'Enhanced','FaceDetectEnhanced')#301

lowLightComparison(790)