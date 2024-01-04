from PIL import Image
Image.MAX_IMAGE_PIXELS = None
def crop_and_resize_image(input_path, output_path, new_size, crop_percentage=0):
    """
    Crop a percentage from all sides of an image and then resize it to the specified size,
    before saving it to the output path.

    :param input_path: Path to the input image.
    :param output_path: Path where the modified image will be saved.
    :param new_size: New size of the image as a tuple (width, height).
    :param crop_percentage: Percentage to crop from each side of the image.
    """
    try:
        with Image.open(input_path) as img:
            if crop_percentage > 0:
                # Calculate the total amount to crop from both sides in each dimension
                crop_width_total = int(img.width * crop_percentage / 100)
                crop_height_total = int(img.height * crop_percentage / 100)

                # Define the left, upper, right, and lower pixel coordinates to crop
                left = crop_width_total
                upper = crop_height_total
                right = img.width - crop_width_total
                lower = img.height - crop_height_total

                # Crop the image
                cropped_img = img.crop((left, upper, right, lower))
            else:
                cropped_img = img

            # Resize the image
            resized_img = cropped_img.resize(new_size)
            resized_img.save(output_path)

    except Exception as e:
        print(f"Error occurred: {e}")

# Example usage:
crop_and_resize_image("class_pos_covid_sparsity_per_class_study.png", "class_pos_covid_sparsity_per_class_study2.png", (8000, 8000), 8)
crop_and_resize_image("class_pos_covid_sparsity_study.png", "class_pos_covid_sparsity_study2.png", (8000, 8000), 8)
