from PIL import Image
import cv2
import numpy as np
import math

def gaussian_filter(sigma,size):   # filter function with input as sigma value
    
    '''
    Function for creating one dim guassian mask
    
    Inputs: 
    Sigma - value of sigma
    Size - size for filter
    
    Output:
    kernel - One gaussian of given size
    '''

    values=int(np.ceil(size*sigma))
    x= np.linspace(-values,values, 2*values+1, dtype=float)     # build a one dim mask
    kernel = 1/(np.sqrt(2*np.pi)*sigma)* np.exp(- x**2/(2*sigma**2))     # Taking it for all the values
    kernel /=np.sum(kernel)
    return kernel



def derv(sigma,size):                  # For derivative along X-axis and Y-axis
    filter=np.array(gaussian_filter(sigma,size))[np.newaxis]  
    return np.dot(filter.T,np.array([[1,0,-1]]))  , np.dot(np.array([[1],[0],[-1]]),filter)      # Using differentiation matrix for derivative 


def convolution(Image,Kernel,stride=1):                    
    
    '''
    Function for Convolution
    
    Inputs: 
    Image - Image 
    Kernel - Gaussian filter kernel
    
    Output:
    output - Resultant Image after convolution 
    '''

    output_shape1=int(((Image.shape[0]-Kernel.shape[0])/stride)+1)
    output_shape2=int(((Image.shape[1]-Kernel.shape[1])/stride)+1)

    output=np.zeros([output_shape1,output_shape2], dtype = int)
    temp=0

    for con_col in range(output_shape1):                       #convolution along the column
        
        for con_row in range(output_shape2):                   #convolution along the row
            
            for row in range(con_col,con_col+Kernel.shape[0]): 
                
                for col in range(con_row, con_row+Kernel.shape[1]):
                    
                    temp+=Image[row][col]*Kernel[row-con_col][col-con_row]
                    
            output[con_col][con_row]=temp

            temp=0

    return output




def padding(Image,pad): 

    output_shape1=Image.shape[0]+pad*2
    output_shape2=Image.shape[1]+pad*2

    output=np.zeros([output_shape1,output_shape2], dtype = int)

    for i in range(pad,output_shape1-pad):

        for j in range(pad,output_shape2-pad):
            
            output[i][j]=Image[i-pad][j-pad]
    
    return output


def col_padding(Image,pad): 

    output=[]

    zeros=np.zeros((1,pad))

    for i in range(len(Image)):  
        temp=np.hstack((Image[i],zeros[0]))
        output.append(temp)
    
    return np.array(output)


def row_padding(Image,pad):
    
    output=[]
    
    zeros=np.zeros((1,Image.shape[1]))
    
    temp=Image
    
    for i in range(pad):
        temp=np.vstack((temp,zeros))
    output.append(temp)

    return np.array(output[0])


def angle_class(direction):             # Classifying directions for thresholding
    
    if direction<0:
        direction+=180

    if  0 <= direction <22.5 :
        return 0

    elif 22.5 <= direction < 67.5 :
        return 45

    elif 67.5 <= direction < 112.5 :
        return 90
        
    elif 112.5 <= direction < 157.5 :
        return 135
    
    elif 157.5 <= direction <=180 :
        return 0



def MagEdgeRes(Image1,Image2):

    '''
    Function for calculating magnitude of Edge response on image
    
    Inputs: 
    Image1 - X component after convolving with derivative
    Image2 - Y component after convolving wiht derivative
    
    Output:
    Mag - Magnitude 
    Direction - Gradient Direction
    '''

    if Image1.shape[0]>Image2.shape[0]:             #Apply padding for having matrices of same shape
        row_pad=Image1.shape[0]-Image2.shape[0]
        Image2=row_padding(Image2,row_pad)
    
    if Image2.shape[0]>Image1.shape[0]:
        row_pad=Image2.shape[0]-Image1.shape[0]
        Image1=row_padding(Image1,row_pad)    
    
    if Image1.shape[1]>Image2.shape[1]:
        col_pad=Image1.shape[1]-Image2.shape[1]
        Image2=col_padding(Image2,col_pad)
    
    if Image2.shape[1]>Image1.shape[1]:
        col_pad=Image2.shape[1]-Image1.shape[1]
        Image1=col_padding(Image1,col_pad)
    
    Mag=np.empty([Image1.shape[0],Image1.shape[1]], dtype = int)
    Direction=np.empty([Image1.shape[0],Image1.shape[1]], dtype = int)
    
    for i in range(0,Image1.shape[0]):
        
        for j in range(0,Image1.shape[1]):
        
            mag=math.sqrt(math.pow(Image1[i][j],2)+math.pow(Image2[i][j],2))            #Calculating magnitude of each pixel
            g_direction=math.atan2(Image2[i][j],Image1[i][j])*180/math.pi               #Calculating gradient direction using atan2()
            g_direction=angle_class(g_direction)                                
            Mag[i][j],Direction[i][j]=mag,g_direction
    
    return Mag,Direction                                                        # Return magnitude and direction to main 





def NMS_filter(Image,Direction):                          
    
    '''
    Function for Non Maximum Supression on image

    Inputs: 
    Image - Magnitude of image
    Direction- Gradient Direction 
    
    Output:
    output - Image after NMS
    '''

    
    Img1=padding(Image,1)                                 #Apply padding for comparison
    D_values=padding(Direction,1)
    
    output=np.zeros([Img1.shape[0],Img1.shape[1]], dtype = int)
    
    for i in range(1,Img1.shape[0]-1):                      # Comparing according to the direction
        for j in range(1,Img1.shape[1]-1):
           
            if D_values[i][j]==0:                       
                if (Img1[i][j] > Img1[i][j-1] and Img1[i][j] > Img1[i][j+1]):         # 0 degrees 
                    output[i][j]=Img1[i][j]
           
            if D_values[i][j]==45:
                if (Img1[i][j] > Img1[i-1][j+1] and Img1[i][j] > Img1[i+1][j-1]):     # 45 degrees 
                    output[i][j]=Img1[i][j]
           
            if D_values[i][j]==90:
                if (Img1[i][j] > Img1[i-1][j] and Img1[i][j] > Img1[i+1][j]):         # 90 degrees 
                    output[i][j]=Img1[i][j]
           
            if D_values[i][j]==135:
                if (Img1[i][j] > Img1[i-1][j-1] and Img1[i][j] > Img1[i+1][j+1]):     # 135 degrees 
                    output[i][j]=Img1[i][j]
    
    return output


def HYS_Threshold(Image):

    '''
    Function for hystersis thresholding on image

    Input: 
    Image - image that you want to apply the hystersis thresholding on
    
    Output:
    Image- image after applying hystersis thresholding
    '''

    high=Image.max()*0.2                # High and low values for thresholding
    low=Image.max()*0.05
    
    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            
            if (Image[i][j]>high):
                Image[i][j]=255

            elif (low< Image[i][j] <= high):
                if ((Image[i-1][j-1]>high) or (Image[i-1][j]>high)       # checking for connectivity of pixels 
                or (Image[i-1][j+1]>high) or (Image[i][j-1]>high) or 
                (Image[i][j+1]>high) or (Image[i+1][j-1]>high) or 
                (Image[i+1][j+1]>high) or (Image[i+1][j]>high)):
                    Image[i][j]=255

            elif (Image[i][j]<low):
                Image[i][j]=0
    
    return Image    


if __name__=='__main__':

    I=Image.open("Input1.jpg")              #reading input image
    
    g=gaussian_filter(0.5,3)                # sigma = 0.5 has best results compared to 3.0 and 5.0 
    g=np.array(g)[np.newaxis]
    
    x=convolution(np.array(I),g)             #Smoothing in x direction
    y=convolution(np.array(I),g.T)           #Smoothing in y direction
    
    gx,gy=derv(0.5,3)                       #derivative of gaussian in x and y directions
    
    xgx_convolve= convolution(x,gx)             #Convolution with derivative for X_Component 
    ygy_convolve= convolution(y,gy)            #Convolution with derivative for Y_Component
    
    mag,direction=MagEdgeRes(xgx_convolve,ygy_convolve)   #For Magnitude and Gradient Direction
    
    nms=NMS_filter(mag,direction)                  #Non Maximum Supression
    
    final=HYS_Threshold(nms)             # Applying Hysteresis for final canny image



    # For sigma = 3.0

    g3=gaussian_filter(3,3)                 
    g3=np.array(g3)[np.newaxis]
    
    x3=convolution(np.array(I),g3)             
    y3=convolution(np.array(I),g3.T)           
    
    gx3,gy3=derv(3,3)                       
    
    xgx_convolve3= convolution(x3,gx3)             
    ygy_convolve3= convolution(y3,gy3)           
    
    mag3,direction3=MagEdgeRes(xgx_convolve3,ygy_convolve3)   
    
    nms3=NMS_filter(mag3,direction3)                  
    
    final3=HYS_Threshold(nms3)



    # For sigma = 5.0
    
    g5=gaussian_filter(5,3)                 
    g5=np.array(g5)[np.newaxis]
    
    x5=convolution(np.array(I),g5)             
    y5=convolution(np.array(I),g5.T)           
    
    gx5,gy5=derv(5,3)                       
    
    xgx_convolve5= convolution(x5,gx5)             
    ygy_convolve5= convolution(y5,gy5)           
    
    mag5,direction5=MagEdgeRes(xgx_convolve5,ygy_convolve5)   
    
    nms5=NMS_filter(mag5,direction5)                  
    
    final5=HYS_Threshold(nms5)
    
    
    #Displaying the results


    
    cv2.imshow("Ix",np.uint8(x))
    cv2.waitKey(0)

    cv2.imshow("Iy",np.uint8(y))
    cv2.waitKey(0)

    cv2.imshow("Ix'",np.uint8(xgx_convolve))
    cv2.waitKey(0)

    cv2.imshow("Iy'",np.uint8(ygy_convolve))
    cv2.waitKey(0)

    cv2.imshow("Magnitude",np.uint8(mag))
    cv2.waitKey(0)

    cv2.imshow("NMS",np.uint8(nms))
    cv2.waitKey(0)

    cv2.imshow("Final",np.uint8(final))
    cv2.waitKey(0)
    
    cv2.imshow("Final sigma = 3",np.uint8(final3))
    cv2.waitKey(0)
   
    cv2.imshow("Final sigma = 5.0",np.uint8(final5))
    cv2.waitKey(0)

    cv2.destroyAllWindows() 

    '''
    For saving  images into your folder

    cv2.imwrite("Ix.jpg",np.uint8(x))
    cv2.imwrite("Iy.jpg",np.uint8(y))
    cv2.imwrite("Ix'.jpg",np.uint8(xgx_convolve))
    cv2.imwrite("Iy'.jpg",np.uint8(ygy_convolve))
    cv2.imwrite("Magnitude.jpg",np.uint8(mag))
    cv2.imwrite("NMS.jpg",np.uint8(nms))
    cv2.imwrite("Final.jpg",np.uint8(final))


    '''
    

    

    

    

    
    

