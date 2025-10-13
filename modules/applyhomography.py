import numpy as np
import numpy.linalg as la
def applyhomography(A,H):
    '''
        Uses bilinear interpolation to transform an input image A according to a
        given 3-by-3 projective transformation matrix H.
        
        Notes:
        
        1. This function follows the (x,y) convention for pixel coordinates,
           which differs from the (row,column) convention. The matrix H must be
           set up accordingly.
        
        2. The size of the output is determined automatically, and the output is
           determined automatically, and the output will contain the entire
           transformed image on a white background. This means that the origin of
           the output image may no longer coincide with the top-left pixel. In
           fact, after executing this function, the true origin (0,0) will be
           located at point (1-minx, 1-miny) in the output image (why?).
        
    '''
    
    # cast the input image to double precision floats
    A = A.astype(float)
    
    # determine number of rows, columns and channels of A
    m, n, c = A.shape
    
    # determine size of output image by forward−transforming the four corners of A
    p1 = np.dot(H,np.array([0,0,1]).reshape((3,1))); p1 = p1/p1[2];
    p2 = np.dot(H,np.array([n-1, 0,1]).reshape((3,1))); p2 = p2/p2[2];
    p3 = np.dot(H,np.array([0, m-1,1]).reshape((3,1))); p3 = p3/p3[2];
    p4 = np.dot(H,np.array([n-1,m-1,1]).reshape((3,1))); p4 = p4/p4[2];
    minx = np.floor(np.amin([p1[0], p2[0], p3[0] ,p4[0]]));
    maxx = np.ceil(np.amax([p1[0], p2[0], p3[0] ,p4[0]]));
    miny = np.floor(np.amin([p1[1], p2[1], p3[1] ,p4[1]]));
    maxy = np.ceil(np.amax([p1[1], p2[1], p3[1] ,p4[1]]));
    nn = int(maxx - minx)
    mm = int(maxy - miny)

    # initialise output with white pixels
    B = np.zeros((mm,nn,c)) + 255

    # pre-compute the inverse of H (we'll be applying that to the pixels in B)
    Hi = la.inv(H)
    
    # Loop  through B's pixels
    for x in range(nn):
        for y in range(mm):
            # compensate for the shift in B's origin
            p = np.array([x + minx, y + miny, 1]).reshape((3,1))
            
            # apply the inverse of H
            pp = np.dot(Hi,p)

            # de-homogenise
            xp = pp[0]/pp[2]
            yp = pp[1]/pp[2]
            
            # perform bilinear interpolation
            xpf = int(np.floor(xp)); xpc = xpf + 1;
            ypf = int(np.floor(yp)); ypc = ypf + 1;

            if ((xpf >= 0) and (xpc < n) and (ypf >= 0) and (ypc < m)):
                B[y,x,:] = (xpc - xp)*(ypc - yp)*A[ypf,xpf,:] \
                            + (xpc - xp)*(yp - ypf)*A[ypc,xpf,:] \
                            + (xp - xpf)*(ypc - yp)*A[ypf,xpc,:] \
                            +  (xp - xpf)*(yp - ypf)*A[ypc,xpc,:] \


    return B.astype(np.uint8)
