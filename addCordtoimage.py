from PIL import Image, ImageDraw, ImageFont

def add_coordinates_to_image(image_path, coordinates, output_path="citymap_with_coords.png"):
    """Adds coordinates to an image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()  # You can specify a different font

    for i, (x, y) in enumerate(coordinates):
        label = f"A{i + 1} ({x}, {y})"
        draw.text((x + 10, y + 10), label, fill=(255, 0, 0), font=font)  # Red text

    img.save(output_path)

# Example usage:
image_path = "citymap.png"
coordinates = [(100, 200), (500, 800), (1200, 300)]  # Your A1, A2, A3 coordinates
add_coordinates_to_image(image_path, coordinates)

print("Coordinates added to citymap_with_coords.png")
