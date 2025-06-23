import numpy as np

# --- Central configuration for synthetic data ---
# These constants are now used by all tests that need synthetic data.
SYNTHETIC_IMG_SIZE = (256, 256)
SYNTHETIC_OBJ_RADIUS_1 = 40
SYNTHETIC_OBJ_CENTER_1 = (70, 70)
SYNTHETIC_OBJ_RADIUS_2 = 30
SYNTHETIC_OBJ_CENTER_2 = (180, 170)

def create_synthetic_data():
    """
    Generates a simple 2D synthetic image and mask in memory using the defined constants.
    """
    img = np.zeros(SYNTHETIC_IMG_SIZE, dtype=np.uint16)
    masks = np.zeros(SYNTHETIC_IMG_SIZE, dtype=np.uint16)

    # Create a grid of coordinates
    rr, cc = np.ogrid[0:SYNTHETIC_IMG_SIZE[0], 0:SYNTHETIC_IMG_SIZE[1]]

    # Circle 1
    c1_y, c1_x = SYNTHETIC_OBJ_CENTER_1
    circle1 = (rr - c1_y)**2 + (cc - c1_x)**2 < SYNTHETIC_OBJ_RADIUS_1**2
    img[circle1] = 500
    masks[circle1] = 1

    # Circle 2
    c2_y, c2_x = SYNTHETIC_OBJ_CENTER_2
    circle2 = (rr - c2_y)**2 + (cc - c2_x)**2 < SYNTHETIC_OBJ_RADIUS_2**2
    img[circle2] = 800
    masks[circle2] = 2

    return img, masks