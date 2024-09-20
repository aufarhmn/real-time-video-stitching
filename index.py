import cv2
import numpy as np

# Function to compute homography based on keypoints
# Returns the homography matrix
def calibrate_homography(frame1, frame2):
    # Initialize ORB detector
    # Detect keypoints and compute descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    # Match descriptors using BFMatcher
    # Sort matches based on distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

# Function to stitch two frames together using homography
# Returns the stitched frame
def stitch_frames(frame1, frame2, homography):
    warped_frame1 = cv2.warpPerspective(frame1, homography, (frame1.shape[1] + frame2.shape[1], frame1.shape[0]))

    warped_frame1[0:frame2.shape[0], 0:frame2.shape[1]] = frame2
    return warped_frame1

# Main function to capture frames from two cameras and stitch them in real-time
# Press 'q' to exit the program
def main():
    cap1 = cv2.VideoCapture(0)  
    cap2 = cv2.VideoCapture(1)  

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret1 and ret2:
        print("Calibrating homography...")
        homography = calibrate_homography(frame1, frame2)
        if homography is None:
            print("Homography calculation failed.")
            return
        print("Homography Matrix Calculated:")
        print(homography)
    else:
        print("Failed to capture frames for calibration")
        return

    print("Starting real-time video stitching...")

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not (ret1 and ret2):
            print("Failed to capture frames from one or both cameras.")
            break

        stitched_frame = stitch_frames(frame1, frame2, homography)

        cv2.imshow("Stitched Video", stitched_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
