import cv2 as cv
import numpy as np
import random
from scipy import linalg # Used for SVD in DLT/Normalization
import matplotlib.pyplot as plt

# ==============================================================================
# HOMOGRAPHY HELPER FUNCTIONS (DLT & Normalization)
# ==============================================================================

def normalize_points(pts):
    """Normalizes 2D points to improve numerical stability for DLT."""
    # Convert to homogeneous coordinates
    homo_pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
    
    # Calculate centroid and translation matrix T_translate
    mean_x, mean_y = np.mean(pts, axis=0)
    T_translate = np.array([[1, 0, -mean_x], [0, 1, -mean_y], [0, 0, 1]])
    
    # Calculate scaling factor 's' to bring average distance to sqrt(2)
    translated_pts = (T_translate @ homo_pts.T).T[:, :2]
    s = np.sqrt(2) / np.mean(np.linalg.norm(translated_pts, axis=1))
    
    # Create scaling matrix T_scale
    T_scale = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
    
    # Final Normalization Matrix T = T_scale * T_translate
    T = T_scale @ T_translate
    
    # Apply full normalization
    normalized_pts = (T @ homo_pts.T).T[:, :2]
    
    return normalized_pts, T

def fit_homography_dlt(p_src, p_dst):
    """
    Fits the Homography (H) matrix using Direct Linear Transform (DLT).
    Requires a minimum of 4 corresponding point pairs.
    """
    if p_src.shape[0] < 4:
        return None
    
    # 1. Normalize points
    p_src_norm, T_src = normalize_points(p_src)
    p_dst_norm, T_dst = normalize_points(p_dst)

    # 2. Construct the matrix A for A*h = 0 (2 equations per point pair)
    A = []
    for i in range(p_src_norm.shape[0]):
        x, y = p_src_norm[i]
        x_prime, y_prime = p_dst_norm[i]
        
        A.append([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y, -x_prime])
        A.append([0, 0, 0, x, y, 1, -y_prime * x, -y_prime * y, -y_prime])
        
    A = np.array(A)
    
    # 3. Solve for h using SVD
    # h is the right singular vector corresponding to the smallest singular value (last row of Vh)
    U, S, Vh = linalg.svd(A)
    H_norm = Vh[-1].reshape(3, 3) 
    
    # 4. Denormalize: H = T_dst^-1 @ H_norm @ T_src
    H = linalg.inv(T_dst) @ H_norm @ T_src
    H = H / H[2, 2] # Normalize by H[2, 2] = 1
    
    return H

# ==============================================================================
# RANSAC IMPLEMENTATION (Custom, based on your structure)
# ==============================================================================

def compute_homography_ransac_custom(src_pts, dst_pts, max_iterations=2000, threshold=5.0):
    """
    Custom RANSAC loop, adapted from your line-fitting structure.
    
    1. Randomly sample P=4 points.
    2. Fit model using DLT (fit_homography_dlt).
    3. Check consensus using Reprojection Error (point_to_line_distance analogue).
    4. Refit model using all inliers.
    """
    if len(src_pts) < 4:
        return None, []

    best_H = None
    max_inliers = 0
    best_inliers_indices = []
    num_points = len(src_pts)

    for i in range(max_iterations):
        # 1. Randomly sample 4 points (P=4 for Homography)
        sample_indices = random.sample(range(num_points), 4)
        sample_src = src_pts[sample_indices]
        sample_dst = dst_pts[sample_indices]
        
        # 2. Fit model (DLT)
        H_candidate = fit_homography_dlt(sample_src, sample_dst)
        
        if H_candidate is None: continue
            
        # 3. Test all points (Reprojection Error)
        src_homo = np.hstack((src_pts, np.ones((num_points, 1))))
        
        # Project source points to destination plane: x' = H * x
        dst_proj_homo = (H_candidate @ src_homo.T).T
        
        # Convert projected points back to Cartesian (divide by w)
        dst_proj = dst_proj_homo[:, :2] / dst_proj_homo[:, 2].reshape(-1, 1)
        
        # Calculate Reprojection Error (Euclidean distance between projected and actual points)
        errors = np.linalg.norm(dst_pts - dst_proj, axis=1)
        
        current_inliers_indices = np.where(errors < threshold)[0]
        current_inlier_count = len(current_inliers_indices)
        
        # 4. Keep best model
        if current_inlier_count > max_inliers:
            max_inliers = current_inlier_count
            best_H = H_candidate
            best_inliers_indices = current_inliers_indices

    # 5. Refit final Homography using all best inliers
    if max_inliers >= 4:
        inlier_src = src_pts[best_inliers_indices]
        inlier_dst = dst_pts[best_inliers_indices]
        final_H = fit_homography_dlt(inlier_src, inlier_dst)
        return final_H, best_inliers_indices
        
    return None, []

# ==============================================================================
# MAIN EXECUTION (Ordered Steps a, b, c)
# ==============================================================================

def execute_stitching(img1_path, img5_path):
    
    print("="*60)
    print(f"STITCHING PROCESS: {img1_path} onto {img5_path}")
    print("="*60)
    
    img1 = cv.imread(img1_path)
    img5 = cv.imread(img5_path)

    if img1 is None or img5 is None:
        print(f"Error: Could not load images. Ensure '{img1_path}' and '{img5_path}' are available.")
        return

    # --------------------------------------------------------------------------
    # (a) Compute and match SIFT features
    # --------------------------------------------------------------------------
    print("--- (a) SIFT Feature Matching ---")
    
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img5_gray = cv.cvtColor(img5, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp5, des5 = sift.detectAndCompute(img5_gray, None)

    flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des5, k=2) 
    
    # Apply Lowe's ratio test (0.75)
    ratio_thresh = 0.75 
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    if len(good_matches) < 4:
        print(f"Error: Only {len(good_matches)} good matches found. Aborting.")
        return

    print(f"  Total good matches found: {len(good_matches)}.")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp5[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    
    # --------------------------------------------------------------------------
    # (b) Compute Homography using your own code within RANSAC and compare
    # --------------------------------------------------------------------------
    print("\n--- (b) Custom RANSAC Homography Computation & Comparison ---")
    
    ransac_threshold = 5.0 
    
    # 2.1 Compute H using CUSTOM RANSAC
    H_custom, inliers_custom_indices = compute_homography_ransac_custom(
        src_pts, dst_pts, threshold=ransac_threshold
    )
    
    if H_custom is None:
        print("  Custom RANSAC failed to find a robust Homography. Aborting.")
        return
        
    print(f"2.1 Custom RANSAC Homography H (Inliers: {len(inliers_custom_indices)}):")
    print(H_custom)

    # 2.2 Compare with the robust benchmark (OpenCV RANSAC)
    H_opencv, mask_opencv = cv.findHomography(src_pts.reshape(-1, 1, 2), 
                                              dst_pts.reshape(-1, 1, 2), 
                                              cv.RANSAC, 
                                              ransac_threshold)
    
    inliers_opencv_count = np.sum(mask_opencv)
    H_diff = H_custom - H_opencv
    frobenius_norm = np.linalg.norm(H_diff, 'fro')
    
    print(f"\n2.2 Robust Benchmark Homography H (OpenCV RANSAC Inliers: {inliers_opencv_count}):")
    print(H_opencv)
    
    print("\n2.3 Comparison:")
    print(f"  Frobenius Norm of Difference (||H_custom - H_benchmark||_F): {frobenius_norm:.4e}")
    
    # --------------------------------------------------------------------------
    # (c) Stitch img1.ppm onto img5.ppm
    # --------------------------------------------------------------------------
    print("\n--- (c) Stitching img1.ppm onto img5.ppm ---")

    H_final_stitch = H_custom 
    h1, w1 = img1.shape[:2]
    h5, w5 = img5.shape[:2]

    # Calculate transformation to fit on canvas
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners1_warped = cv.perspectiveTransform(corners1, H_final_stitch)

    all_corners = np.concatenate((corners1_warped, np.float32([[0, 0], [0, h5], [w5, h5], [w5, 0]]).reshape(-1, 1, 2)), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation_x, translation_y = -x_min, -y_min

    H_translation = np.array([[1, 0, translation_x], [0, 1, translation_y], [0, 0, 1]], dtype=np.float32)
    H_stitch = H_translation @ H_final_stitch

    output_w = x_max - x_min
    output_h = y_max - y_min

    # Warp img1
    stitched_img = cv.warpPerspective(img1, H_stitch, (output_w, output_h))

    # Place img5 (base image) onto the canvas
    stitched_img[translation_y:h5 + translation_y, translation_x:w5 + translation_x] = img5

    # Save final image
    output_filename = 'stitched_graffiti_custom_ransac.jpg'
    cv.imwrite(output_filename, stitched_img)

    print(f"\nStitching complete. Result saved as '{output_filename}'.")
    print("="*60)


# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == '__main__':
    IMG1_PATH = 'C:/Users/Luchitha/Documents/Python/computer vision/images2/graf/img1.ppm'
    IMG5_PATH = 'C:/Users/Luchitha/Documents/Python/computer vision/images2/graf/img5.ppm'

    execute_stitching(IMG1_PATH, IMG5_PATH)