import pygetwindow as gw
import pyautogui
import re
import matplotlib.pyplot as plt
import numpy as np
import pydirectinput
from PIL import Image, ImageDraw
import cv2
import numpy as np
import math



def remove_pixels_right_of_color(img, target_color):
    pixels = img.load()  # Load the pixel data
    width, height = img.size

    # Iterate through each pixel
    for y in range(height):
        remove = False
        for x in range(width):
            if pixels[x, y] == target_color:
                remove = True  # Found the target color, start removing pixels to the right
            if remove:
                # Set to white or any other color, modify this as needed
                pixels[x, y] = (0, 0, 0)

    return img  # Return the modified image

def set_pixel_to_same_color_as_pixel_over_or_under_if_black(img):
    pixels = img.load()  # Load the pixel data
    width, height = img.size

    # Iterate through each pixel
    for y in range(height):
        for x in range(width):
            if pixels[x, y] == (0, 0, 0):
                # Set to the color of the pixel above or below
                if y > 0:
                    pixels[x, y] = pixels[x, y - 1]
                elif y < height - 1:
                    pixels[x, y] = pixels[x, y + 1]

    return img  # Return the modified image

def over_half_of_pixels_are_black(img):
    pixels = img.load()  # Load the pixel data
    width, height = img.size

    # Iterate through each pixel
    black_pixels = 0
    for y in range(height):
        for x in range(width):
            if pixels[x, y] == (0, 0, 0):
                black_pixels += 1

    return black_pixels / (width * height) > 0.5

def crop_image_to_width_from_right(img, width):
    img = img.crop((img.width - width, 0, img.width, img.height))
    return img


def crop_black_areas2(imageIn, black_threshold=1):
    # Convert PIL Image to NumPy array
    image = np.array(imageIn)
    # Convert RGB to BGR (OpenCV uses BGR color space)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Create a mask where non-black pixels are marked as white
    mask = cv2.inRange(image, (0, 0, 0), (black_threshold, black_threshold, black_threshold))

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return original image
    if not contours:
        return image

    # Find the bounding rectangle for each contour
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Crop the original image with the bounding rectangle of the largest contour
    cropped_image = image[y:y+h, x:x+w]
        # Convert the NumPy array back to a PIL Image
    cropped_image_pil = Image.fromarray(cropped_image)

    return cropped_image_pil


def crop_black_areas(imageIn, black_threshold=1):
    # Convert PIL Image to NumPy array
    image = np.array(imageIn)
    # Convert RGB to BGR (OpenCV uses BGR color space)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mask = cv2.inRange(image, (0, 0, 0), (black_threshold, black_threshold, black_threshold))

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return original image as PIL Image
    if not contours:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Calculate the combined bounding box of all contours
    x_min, y_min = np.min([cv2.boundingRect(c)[:2] for c in contours], axis=0)
    x_max, y_max = np.max([cv2.boundingRect(c)[0]+cv2.boundingRect(c)[2] for c in contours],
                          axis=0), np.max([cv2.boundingRect(c)[1]+cv2.boundingRect(c)[3] for c in contours], axis=0)

    # Crop the original image with the combined bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Convert the NumPy array back to a PIL Image
    cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    return cropped_image_pil


def remove_black_pixels_pil(img):
    """
    This function removes all black pixels from a PIL Image object.

    Args:
    img (PIL.Image): A PIL Image object.

    Returns:
    PIL.Image: An image object with all black pixels turned transparent.
    """
    # Convert the image to RGBA (if not already in that format)
    img = img.convert("RGBA")

    # Load the data of the image
    data = img.getdata()

    # Create new data without black pixels
    new_data = []
    for item in data:
        # Change all black (also nearly black if you want) pixels to transparent
        if item[0] == 0 and item[1] == 0 and item[2] == 0:  # RGB values are all 0
            new_data.append((255, 255, 255, 0))  # Full transparency
        else:
            new_data.append(item)

    # Update image data
    img.putdata(new_data)
    return img

def crop_transparent_background_pil(img):
    """
    This function takes a PIL Image object, removes the transparent background by cropping it out,
    and returns the resulting PIL Image object.
    """
    # Convert the image to RGBA if it is not already
    img = img.convert("RGBA")

    # Get the bounding box
    bbox = img.getbbox()

    # Crop the image to the contents and return
    if bbox:
        cropped_img = img.crop(bbox)
        return cropped_img
    else:
        # If the image is completely transparent
        print("The image is completely transparent.")
        return None

def add_center_circle_to_pil(img, circle_radius):
    # Make a copy of the image to avoid modifying the original one
    new_img = img.copy()
    draw = ImageDraw.Draw(new_img)

    # Calculate the position for the circle (center of the image)
    img_width, img_height = new_img.size
    x = (img_width - circle_radius) / 2
    y = (img_height - circle_radius) / 2

    # Draw the circle
    draw.ellipse((x, y, x + circle_radius, y + circle_radius), fill='black')

    # Return the modified image
    return new_img



def blacken_outside_circle(pil_img, radius):
    # Ensure the radius does not exceed the size of the image
    radius = min(radius, min(pil_img.size) // 2)

    # Create a new image with the same size as the original one, filled with black color
    black_img = Image.new('RGB', pil_img.size, 'black')

    # Create a mask (same size as the original image) with a white-filled circle in the center
    mask = Image.new('L', pil_img.size, 0)  # 'L' mode for grayscale
    draw = ImageDraw.Draw(mask)
    center = (pil_img.size[0] // 2, pil_img.size[1] // 2)  # (x, y) center of the image
    draw.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius], fill=255)

    # Combine the original image and the black image using the mask
    # Pixels outside the circle in the mask are black (from black_img), those inside are from the original image
    result = Image.composite(pil_img, black_img, mask)
    return result


def find_strictly_red_region(pil_img, grid_size=10, red_threshold=50, dominance_factor=2, min_red_pixels=1):
    """
    Find the region with the most distinctly red pixels in the PIL Image.
    Returns None if no region meets the required number of distinctly red pixels.

    Args:
    pil_img (PIL.Image): The image to analyze.
    grid_size (int): The size of the grid to divide the image into, in pixels.
    red_threshold (int): The minimum value for the red component to be considered red.
    dominance_factor (int): The factor by which red must exceed each of the other two colors.
    min_red_pixels (int): The minimum number of red pixels required in a region for it to be considered.

    Returns:
    tuple or None: The top-left and bottom-right coordinates of the grid square with the required number of distinctly red pixels,
                   or None if no such region is found.
    """
    # Convert the image to RGB if it's not already.
    img = pil_img.convert('RGB')
    width, height = img.size

    # Initialize variables to store the maximum red value found and its coordinates.
    max_red_count = 0
    max_coords = None

    # Iterate over the image in blocks.
    for x in range(0, width, grid_size):
        for y in range(0, height, grid_size):
            # Initialize the count of distinctly red pixels in this block.
            red_count = 0
            # Iterate over the pixels in the block.
            for i in range(x, min(x + grid_size, width)):
                for j in range(y, min(y + grid_size, height)):
                    # Check if the pixel is distinctly red.
                    r, g, b = img.getpixel((i, j))
                    if r > red_threshold and r > dominance_factor * max(g, b):
                        red_count += 1

            # Update the maximum and coordinates if this block has more distinctly red pixels than the current maximum
            # and exceeds the minimum number of red pixels required.
            if red_count > max_red_count and red_count >= min_red_pixels:
                max_red_count = red_count
                max_coords = (x, y, x + grid_size, y + grid_size)

    # Return the coordinates of the block with the required number of distinctly red pixels, or None if no such block was found.
    return max_coords


def draw_blue_box(image, max_coords):
    """
    Draws a blue box on the provided PIL image.

    Args:
    image (PIL.Image): The image on which to draw the box.
    max_coords (tuple): A tuple containing the coordinates (x, y, x + grid_size, y + grid_size) for the blue box.

    Returns:
    PIL.Image: The modified image with the blue box drawn on it.
    """
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    # Define the box color
    box_color = 'blue'
    # Draw the rectangle
    draw.rectangle(max_coords, outline=box_color, width=3)
    # Return the modified image
    return image



def are_coords_close(cord1, cord2, threshold=10):
    # Extract individual coordinates
    x1, y1, x2, y2 = cord1
    a1, b1, a2, b2 = cord2

    # Calculate center points of the coordinates
    center_x1, center_y1 = (x1 + x2) / 2, (y1 + y2) / 2
    center_a1, center_b1 = (a1 + a2) / 2, (b1 + b2) / 2

    # Calculating the distance between the centers of two grid squares (using Pythagorean theorem)
    distance = math.sqrt((center_x1 - center_a1) ** 2 + (center_y1 - center_b1) ** 2)

    # Compare with threshold
    return distance <= threshold

lastCords = (0, 0, 0, 0)

def main():
    global lastCords

    # Get all windows with 'RuneLite' in the title
    runelite_windows = [win for win in gw.getWindowsWithTitle('RuneLite') if 'RuneLite' in win.title]

    for window in runelite_windows:
        try:

            # if the window is not active, skip it
            if not window.isActive:
                continue

            # Select from top 300px, from right 500px of the window
            width, height = 515, 262
            region = (window.left + window.width - width, window.top, width, height)





            screenshot = pyautogui.screenshot(region=region)

            screenshot.save('maot_modified.png')
            modified_image = remove_pixels_right_of_color(screenshot, (40, 40, 40))  # Modify the image
            modified_image.save('modified_1.png')
            modified_image = remove_pixels_right_of_color(screenshot, (47, 47, 47))  # Modify the image
            modified_image.save('modified_2.png')
            modified_image = remove_pixels_right_of_color(modified_image, (30, 30, 30))  # Modify the image
            modified_image.save('modified_3.png')
            set_pixel_to_same_color_as_pixel_over_or_under_if_black(modified_image)  # Modify the image
            modified_image.save('modified_4__.png')
            is_extended = over_half_of_pixels_are_black(modified_image)  # Check if the image is extended

            if is_extended == False:
                modified_image = crop_image_to_width_from_right(modified_image, 240)
                modified_image.save('modified_4_5.png')

            # modified_image = crop_black_areas(modified_image)
            modified_image = remove_black_pixels_pil(modified_image)
            modified_image.save('modified_5.png')
            modified_image = crop_transparent_background_pil(modified_image)
            modified_image.save('modified_6.png')

            modified_image = add_center_circle_to_pil(modified_image, 200)
            modified_image.save('modified_7.png')
            modified_image = blacken_outside_circle(modified_image, 103)
            modified_image.save('modified_8.png')

            cords = find_strictly_red_region(modified_image, min_red_pixels=10)

            print(cords)

            if cords == None:
                continue

            if lastCords != None and lastCords != (0, 0, 0, 0):
                if are_coords_close(cords, lastCords, 200):
                    continue

            lastCords = cords


            preview_image = draw_blue_box(modified_image, cords)
            preview_image.save('preview_1.png')


            # modified_image.save('modified.png')

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    lastCords = (0, 0, 0, 0)
    while True:
        main()
        # Sleep for 100ms to avoid using too much CPU
        pyautogui.sleep(0.3)


