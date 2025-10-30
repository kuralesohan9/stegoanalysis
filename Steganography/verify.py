import cv2
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Verify the difference between an original and a stego image.")
    parser.add_argument("--original", required=True, help="Path to the original cover image.")
    parser.add_argument("--stego", required=True, help="Path to the stego image.")
    parser.add_argument("--output", default="difference_map.png", help="Path to save the difference map image.")
    
    args = parser.parse_args()

    print("Loading images...")
    original_img = cv2.imread(args.original)
    stego_img = cv2.imread(args.stego)

    if original_img is None or stego_img is None:
        print("Error: Could not load one or both images. Check file paths.")
        return

    if original_img.shape != stego_img.shape:
        print("Error: Images have different dimensions. Resizing stego image for comparison.")
        stego_img = cv2.resize(stego_img, (original_img.shape[1], original_img.shape[0]))

    print("Calculating the difference between the images...")
    # Compute the absolute difference
    difference = cv2.absdiff(original_img, stego_img)
    
    # Count the number of pixels that are not black (i.e., were changed)
    changed_pixels = np.count_nonzero(np.all(difference > 0, axis=2))
    
    if changed_pixels == 0:
        print("\n--- Verification Result ---")
        print("Result: The images are identical. No data was embedded.")
    else:
        print("\n--- Verification Result ---")
        print(f"Result: Found {changed_pixels} pixels that were modified.")
        print("This confirms that data has been embedded in the stego image.")
        
        # To make the difference visible, we can scale the values
        difference_visual = difference * 50 # Amplify changes to make them easier to see
        cv2.imwrite(args.output, difference_visual)
        print(f"A visual 'difference map' has been saved to: {args.output}")

if __name__ == "__main__":
    main()