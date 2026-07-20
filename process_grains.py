import numpy as np
import pandas as pd
import os
from skimage import io, color, filters, morphology, measure, transform

KNOWN_DIAMETER_MM = 30.0
DATASET_FOLDER = 'dataset'
OUTPUT_FOLDER = 'processed_grains'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def rotate_grain_to_vertical(grain_crop, orientation_rad):
    angle_deg = -np.rad2deg(orientation_rad)
    rotated = transform.rotate(grain_crop, angle_deg, resize=True, preserve_range=True)
    return rotated.astype(np.uint8)

image_files = sorted([f for f in os.listdir(DATASET_FOLDER) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

combined_metadata = []

for image_file in image_files:
    print(f"Processing image: {image_file}")
    img_path = os.path.join(DATASET_FOLDER, image_file)
    img = io.imread(img_path)
    gray = color.rgb2gray(img)
    
    thresh = filters.threshold_otsu(gray)
    binary = gray > thresh
    binary = morphology.remove_small_objects(binary, min_size=50)
    binary = morphology.remove_small_holes(binary, area_threshold=50)
    binary = morphology.binary_closing(binary, morphology.disk(2))
    
    label_img = measure.label(binary)
    regions = measure.regionprops(label_img)
    if not regions:
        print(f"No grains detected in {image_file}")
        continue
    
    ref_object = min(regions, key=lambda r: (r.eccentricity, -r.area))
    pixel_diameter = ref_object.equivalent_diameter
    mm_per_pixel = KNOWN_DIAMETER_MM / pixel_diameter
    
    base_name = os.path.splitext(image_file)[0]
    grain_count = 1
    
    for region in regions:
        if region.label == ref_object.label:
            continue
        
        minr, minc, maxr, maxc = region.bbox
        m = 10
        grain_crop = img[
            max(0, minr-m):min(img.shape[0], maxr+m),
            max(0, minc-m):min(img.shape[1], maxc+m)
        ]
        
        rotated_crop = rotate_grain_to_vertical(grain_crop, region.orientation)
        height_mm = region.major_axis_length * mm_per_pixel
        width_mm = region.minor_axis_length * mm_per_pixel
        
        mask = (label_img == region.label)[minr:maxr, minc:maxc]
        grain_pixels = img[minr:maxr, minc:maxc][mask]
        avg_color = np.mean(grain_pixels, axis=0) if grain_pixels.size > 0 else [0, 0, 0]
        
        grain_filename = f"{base_name}_Grain_{grain_count:04d}.jpg"
        io.imsave(os.path.join(OUTPUT_FOLDER, grain_filename), rotated_crop)
        
        rice_type = image_file.split('_')[1] if '_' in image_file else "Unknown"
        
        combined_metadata.append({
            "Grain Image Name": grain_filename,
            "Rice Type": rice_type,
            "Height (mm)": round(height_mm, 4),
            "Width (mm)": round(width_mm, 4),
            "Average Color": avg_color.tolist()
        })
        
        grain_count += 1

combined_df = pd.DataFrame(combined_metadata)
combined_csv_path = os.path.join(OUTPUT_FOLDER, 'metadata.csv')
combined_df.to_csv(combined_csv_path, index=False)

print(f"All processing done! CSV saved at: {combined_csv_path}")
print(f"Total grains processed: {len(combined_df)}")
