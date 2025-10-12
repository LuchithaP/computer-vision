import cv2
import numpy as np
import matplotlib.pyplot as plt



def select_points_matplotlib(image_rgb, num_points=4, title=None):
    """
    Display `image_rgb` using matplotlib and let the user click `num_points` times.
    Returns list of (x, y) integer coordinates in image pixel space.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_rgb)
    ax.set_title(title or f"Click {num_points} points (close window when done)")
    ax.axis('off')

    # Use ginput for simplicity: this will block until num_points clicks are made
    pts = plt.ginput(num_points, timeout=0)  # timeout=0 -> wait indefinitely
    plt.close(fig)

    if len(pts) != num_points:
        raise RuntimeError(f"Expected {num_points} points, got {len(pts)}")

    # ginput returns (x, y) floats; convert to (int(x), int(y))
    pts_int = [(int(round(x)), int(round(y))) for (x, y) in pts]
    return pts_int



def click_points(event, x, y, flags, param):
    global points, img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Target Image', img)
        if len(points) == 4:
            cv2.destroyAllWindows()


def compute_homography_and_warp(target_img, source_img, dst_pts):
    """
    Compute homography from full source corners to dst_pts and warp the source
    to the target image size. Returns (H, warped, mask).
    """
    h_src, w_src = source_img.shape[:2]
    src_pts = np.array([
        [0, 0],
        [w_src - 1, 0],
        [w_src - 1, h_src - 1],
        [0, h_src - 1]
    ], dtype=np.float32)

    dst_pts_arr = np.array(dst_pts, dtype=np.float32)

    H, status = cv2.findHomography(src_pts, dst_pts_arr, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Could not compute homography")

    h_tgt, w_tgt = target_img.shape[:2]
    warped = cv2.warpPerspective(source_img, H, (w_tgt, h_tgt))

    mask = np.zeros((h_tgt, w_tgt), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_pts_arr), 255)

    return H, warped, mask


def blend_images(target_img, warped_img, mask, blend_alpha=1.0, blur_ksize=7):
    """
    Blend warped_img into target_img using mask and optional alpha. Returns BGR image.
    """
    if blend_alpha <= 0:
        return target_img.copy()

    # smooth mask edges
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    mask_blurred = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    mask_3c = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    tgt_f = target_img.astype(np.float32)
    wrp_f = warped_img.astype(np.float32)

    blended = (tgt_f * (1 - blend_alpha * mask_3c) + wrp_f * (blend_alpha * mask_3c)).astype(np.uint8)
    return blended


def superimpose_images(target_img, source_img, dst_points=None, output_path=None, blend_alpha=1.0,
                       use_matplotlib_selector=True):
    """
    Superimpose source_img onto target_img using homography. If dst_points is None,
    opens an interactive selector (matplotlib) to collect 4 points.

    Parameters
    ----------
    target_img, source_img : np.ndarray (BGR)
    dst_points : list of 4 (x,y) tuples in pixel coordinates in the target image
    output_path : optional file path to save result
    blend_alpha : 0..1 transparency
    use_matplotlib_selector : if True and dst_points is None, use plt.ginput to get points

    Returns
    -------
    blended (BGR), warped (BGR), mask (single channel uint8), H (3x3 homography)
    """
    if target_img is None or source_img is None:
        raise ValueError("Input images must not be None")

    # Prepare RGB copy for display if selecting points
    if dst_points is None and use_matplotlib_selector:
        tgt_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        pts = select_points_matplotlib(tgt_rgb, num_points=4,
                                       title='Select 4 corners: TL, TR, BR, BL')
    elif dst_points is None:
        raise ValueError("dst_points must be provided when not using interactive selector")
    else:
        pts = dst_points

    # Compute homography and warp
    H, warped, mask = compute_homography_and_warp(target_img, source_img, pts)

    # Blend
    blended = blend_images(target_img, warped, mask, blend_alpha=blend_alpha)

    if output_path:
        cv2.imwrite(output_path, blended)

    return blended, warped, mask, H

def demo(target_path, source_path, output_path=None, blend_alpha=1.0):
    

    target = cv2.imread(target_path)
    source = cv2.imread(source_path)

    if target is None or source is None:
        raise RuntimeError("Could not load images. Check file formats and paths.")

    print("Displaying target image — click 4 points: TL, TR, BR, BL")
    blended, warped, mask, H = superimpose_images(target, source, output_path=output_path,
                                                  blend_alpha=blend_alpha,
                                                  use_matplotlib_selector=True)

    # Show result inline using matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Target')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Warped Source')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Blended Result')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    if output_path:
        print(f"Saved result to: {output_path}")
    else:
        print("Done — no output file was written.")
        
source_path = "C:/Users/Luchitha/Documents/Python/computer vision/Assignment2/images/ronaldo.jpg"
target_path = "C:/Users/Luchitha/Documents/Python/computer vision/Assignment2/images/wall.jpg"
output_path = "C:/Users/Luchitha/Documents/Python/computer vision/Assignment2/images/output.jpg"

target_path_1 = "C:/Users/Luchitha/Documents/Python/computer vision/Assignment2/images/uom_front.png"
source_path_1 = "C:/Users/Luchitha/Documents/Python/computer vision/Assignment2/images/uom_logo_1.jpeg"
output_path_1 = "C:/Users/Luchitha/Documents/Python/computer vision/Assignment2/images/output1.jpg"

demo(target_path, source_path, output_path, blend_alpha=0.7)
