"""
    A script to hand draw mask on images.
    1. Load and display an image.
    2. Register click event to draw mask, press 'q' to quit.
    3. Save mask to file.
"""

import cv2
import numpy as np

def on_key_press(key):
    global radius, last_mask, mask, img
    if key == ord('q'):
        # do something if 'q' key is pressed
        pass
    elif key == ord('s'):
        # do something if 's' key is pressed
        cv2.imwrite('mask.png', mask)
    elif key == ord('i'):
        radius += 1
        print(f"radius: {radius}")
    elif key == ord('k'):
        radius -= 1
        print(f"radius: {radius}")
    elif key == ord('z'):
        mask = last_mask.copy()
        canvas = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow('image', canvas)

# Define a callback function for mouse events
def draw_rect(event, x, y, flags, param):
    global last_mask, mask, img, radius
    # If the left mouse button is pressed, record the starting position of the rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        draw_rect.start_x = x
        draw_rect.start_y = y
    # If the left mouse button is released, draw the rectangle on the image
    elif event == cv2.EVENT_LBUTTONUP:
        displacement = np.array([x - draw_rect.start_x, y - draw_rect.start_y])
        """
            Draw a list of circle along the line
        """
        last_mask = mask.copy()
        if np.linalg.norm(displacement) < radius:
            center = np.array([x, y])
            cv2.circle(mask, tuple(center.astype(np.int)), radius, 0, -1)
        else:
            # Get the direction of the line
            direction = displacement / np.linalg.norm(displacement)
            # Get the length of the line
            length = np.linalg.norm(displacement)
            # Get the number of circles
            num_circles = int(length / (radius/2))
            # Get the radius of the circles
            # Get the center of the circles
            centers = np.array([draw_rect.start_x, draw_rect.start_y]) + np.arange(num_circles)[:, None] * direction * (radius / 2)
            # Draw the circles
            for center in centers:
                cv2.circle(mask, tuple(center.astype(np.int)), radius, 0, -1)

        canvas = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow('image', canvas)


if __name__ == '__main__':
    img = cv2.imread('/home/yxliu/multi_cam/monodepth/0000000003.png')

    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    canvas = img.copy()
    mask = np.ones(img.shape[:2], dtype=np.uint8)
    last_mask = mask.copy()
    radius = 10
    # Create a window to display the image
    cv2.namedWindow('image')

    # Set the mouse callback function for the window
    cv2.setMouseCallback('image', draw_rect)

    # Display the image
    cv2.imshow('image', canvas)

    # Wait for the user to press a key
    try:
        while True:
            key = cv2.waitKey(1)
            if key != -1:
                on_key_press(key)
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Close the window
        cv2.destroyAllWindows()
