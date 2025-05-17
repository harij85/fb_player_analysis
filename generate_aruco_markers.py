import cv2
import os

def generate_aruco_marker(marker_id, size=400, save_dir='markers'):
    """
    Generate and save an ArUco marker image.

    Args:
        marker_id (int): ID of the marker (0–49 for DICT_4X4_50).
        size (int): Pixel width/height of the marker.
        save_dir (str): Directory to save the marker PNG.
    """
    os.makedirs(save_dir, exist_ok=True)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)

    filename = os.path.join(save_dir, f"marker_{marker_id}.png")
    cv2.imwrite(filename, marker_image)
    print(f"✅ Saved ArUco marker ID {marker_id} to {filename}")

if __name__ == '__main__':
    # Generate a few markers for demo
    for i in range(14):  # Change this to however many you need
        generate_aruco_marker(marker_id=i, size=3600)
