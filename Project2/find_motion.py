import numpy as np
import cv2

# 主要函式 要找motion model
def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    f1 -- list of cv2.KeyPoint objects in the first image
    f2 -- list of cv2.KeyPoint objects in the second image
    matches -- list of cv2.DMatch objects
        DMatch.queryIdx: The index of the feature in the first image
        DMatch.trainIdx: The index of the feature in the second image
        DMatch.distance: The distance between the two features
    (應該就是前面的一些結構)
    
    m: 哪種 motion model  (如果只有translation就給0)
    matches: matching list
    nRANSAC: ransac要跑幾次
    RANSACthresh: 距離多近才算inlier (可能要先跑跑看才知道thresh要設多少)

    回傳一個 估計出來的motion model (3x3 np矩陣)
    '''

    if m == 0:  # 代表只有translation
        s = 1
    if m == 1:  # 應該是affine
        s = 4

    num_inliers = -1
    best_estimate = []

    for _ in range(nRANSAC):
        # 隨機選一個pair
        matches_sub = np.random.choice(matches, s, replace=False)
        if m == 1:
            H = computeHomography(f1, f2, matches_sub)
        if m == 0:
            x1, y1 = f1[matches_sub[0].queryIdx].pt
            x2, y2 = f2[matches_sub[0].trainIdx].pt
            # 只是translation的話就直接設定就好
            H = np.zeros((3, 3))
            H[0, 0] = 1
            H[1, 1] = 1
            H[2, 2] = 1
            H[0, 2] = x2 - x1
            H[1, 2] = y2 - y1
        # 找 inlier 們
        inlier_indices = getInliers(f1, f2, matches, H, RANSACthresh)

        # 如果 inlier們的數量比較多 代表是一個比較好的model
        # 用 best_estimate存inlier們
        if (len(inlier_indices) > num_inliers):
            num_inliers = len(inlier_indices)
            best_estimate = inlier_indices

    # 用inlier們算出motion model
    M = leastSquaresFit(f1, f2, matches, m, best_estimate)

    return M


# 給定可能的motion model，找inlier們
def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
        M -- inter-image transformation matrix (3x3)
        RANSACthresh -- RANSAC distance threshold
    Output: indices of in-liers in matches (inlier們是指某些配對們)
    '''

    # 對於每個pair，去看這個pair是不是同意給定的motion model(M)

    inlier_indices = []

    for i in range(len(matches)):

        x_f1,y_f1 = f1[matches[i].queryIdx].pt
        x_f2,y_f2 = f2[matches[i].trainIdx].pt

        mat_x = np.zeros((3,1))

        # 2d -> 3d 才能夠做translation
        mat_x[0] = x_f1
        mat_x[1] = y_f1
        mat_x[2] = 1
        
        # 轉過去的對應點
        y = np.dot(M, mat_x)

        if (np.sqrt( ((y[0]/y[2])-x_f2)**2 + ((y[1]/y[2])-y_f2)**2 ) <= RANSACthresh):
            inlier_indices.append(i)

    return inlier_indices


def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')
    Output:
        M - transformation matrix
        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # 3d 的 單位矩陣
    M = np.eye(3)

    # 他定的常數應該是這樣
    eTranslate = 0
    eHomography = 1
    
    if m == eTranslate:
        # cyy上課講過就取平均就好

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):

            x_f1,y_f1 = f1[matches[inlier_indices[i]].queryIdx].pt
            x_f2,y_f2 = f2[matches[inlier_indices[i]].trainIdx].pt
            u = u + x_f2 - x_f1
            v = v + y_f2 - y_f1

        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0,2] = u
        M[1,2] = v

    elif m == eHomography:  # 是translation就用不到
        best = []
        inlier_length = len(inlier_indices)
        for i in range(inlier_length):
            best.append(matches[inlier_indices[i]])
        M = computeHomography(f1, f2, best)

    return M


###############################################################################
# 如果是tranlation的話不需要這個
# 應該就只是在算affine之類的 比較複雜的motion model
def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        #BEGIN TODO 2
        #Fill in the matrix A in this loop.
        #Access elements using square brackets. e.g. A[0,0]
        #TODO-BLOCK-BEGIN

        A[i*2,0] = a_x
        A[i*2,1] = a_y
        A[i*2,2] = 1
        A[i*2,6] = -b_x * a_x
        A[i*2,7] = -b_x * a_y
        A[i*2,8] = -b_x

        A[i*2 + 1,3] = a_x
        A[i*2 + 1,4] = a_y
        A[i*2 + 1,5] = 1
        A[i*2 + 1,6] = -b_y * a_x
        A[i*2 + 1,7] = -b_y * a_y
        A[i*2 + 1,8] = -b_y

        #TODO-BLOCK-END
        #END TODO

    U, s, Vt = np.linalg.svd(A)

    if A_out is not None:
        A_out[:] = A

    #s is a 1-D array of singular values sorted in descending order
    #U, Vt are unitary matrices
    #Rows of Vt are the eigenvectors of A^TA.
    #Columns of U are the eigenvectors of AA^T.

    #Homography to be calculated
    H = np.eye(3)

    #BEGIN TODO 3
    #Fill the homography H with the appropriate elements of the SVD
    #TODO-BLOCK-BEGIN

    #SLIDE 7 IN VISUALIZED STEP PPT

    eig_shape = Vt.shape
    h = Vt[eig_shape[0]-1] #LAST ROW, SMALLEST EIGENVALUE
    H_len = len(H)
    for i in range(H_len):
        for j in range(H_len):
            H[i, j] = h[H_len * i + j]

    #TODO-BLOCK-END
    #END TODO

    return H
