import os
import numpy as np
import tifffile as tiff
import open3d as o3d
from pathlib import Path
from PIL import Image
import math
import mvtec3d_util as mvt_util
import argparse
import render_utils
import glob
import shutil
import cv2
import time

def process_pointcloud_data(point_cloud, color_img, target_width, target_height):
    """
    Process point cloud data and resize both point cloud and RGB image to specified dimensions
    
    Args:
        point_cloud: Organized point cloud data
        color_img: RGB image corresponding to point cloud
        target_width: Target width for resizing
        target_height: Target height for resizing
        
    Returns:
        Tuple containing processed point cloud, resized RGB image, and non-zero RGB values
    """
    # Resize point cloud and image to target dimensions
    resized_pc = mvt_util.resize_organized_pc(point_cloud, 
                                            target_height=target_height, 
                                            target_width=target_width, 
                                            tensor_out=False)
    resized_img = cv2.resize(color_img, (target_height, target_width))
    
    # Convert to unorganized point cloud and remove zero points
    unorg_pc = mvt_util.organized_pc_to_unorganized_pc(resized_pc)
    valid_indices = np.nonzero(np.all(unorg_pc != 0, axis=1))[0]
    clean_pc = unorg_pc[valid_indices, :]
    
    # Create Open3D point cloud and transform coordinate system
    o3d_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(clean_pc))
    o3d_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    # Process and extract valid RGB values
    rgb_values = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    rgb_values = rgb_values.reshape(resized_img.shape[0] * resized_img.shape[1], 
                                   resized_img.shape[2])
    rgb_values = rgb_values[valid_indices, :] / 255.
    
    return o3d_cloud, resized_img, rgb_values

def generate_multiview_data(renderer, input_path, source_root, output_dir):
    """
    Generate multi-view RGB images from 3D point cloud data
    
    Args:
        renderer: Multi-view renderer instance
        input_path: Path to point cloud TIFF file
        source_root: Root directory of source dataset
        output_dir: Directory to save processed data
    """
    # Define file paths
    color_img_path = str(input_path).replace("xyz", "rgb").replace("tiff", "png")
    mask_path = str(input_path).replace("xyz", "gt").replace("tiff", "png")
    has_mask = os.path.isfile(mask_path)
    
    # Read and process point cloud and RGB image
    point_cloud = mvt_util.read_tiff_organized_pc(input_path)
    color_image = cv2.imread(color_img_path)
    
    # Process at 224x224 resolution and original resolution
    pc_224, img_224, rgb_224 = process_pointcloud_data(point_cloud, color_image, 224, 224)
    pc_original, img_original, rgb_original = process_pointcloud_data(point_cloud, color_image, 
                                                                     color_image.shape[0], 
                                                                     color_image.shape[1])
    
    # Calculate features and render views
    _ = renderer.calculate_fpfh_features(pc_original)
    rendered_views, view_points = renderer.multiview_render(pc_original, rgb_original, pc_224)
    features = renderer.calculate_fpfh_features(pc_224)
    
    # Prepare output paths
    output_rgb_path = color_img_path.replace(source_root, output_dir)
    output_mask_path = mask_path.replace(source_root, output_dir)
    output_rgb_dir = os.path.split(output_rgb_path)[0]
    output_mask_dir = os.path.split(output_mask_path)[0]
    
    tiff_path_split = os.path.split(input_path.replace(source_root, output_dir))
    pc_output_dir = os.path.join(tiff_path_split[0], tiff_path_split[1][:-5])
    pc_output_path = os.path.join(pc_output_dir, 'xyz.tiff')
    
    # Create output directories
    os.makedirs(pc_output_dir, exist_ok=True)
    os.makedirs(output_rgb_dir, exist_ok=True)
    if has_mask:
        os.makedirs(output_mask_dir, exist_ok=True)
    
    # Save processed data
    cv2.imwrite(output_rgb_path, img_224)
    if has_mask:
        mask_image = cv2.imread(mask_path)
        mask_image = cv2.resize(mask_image, (224, 224))
        cv2.imwrite(output_mask_path, mask_image)
    
    shutil.copy(input_path, pc_output_path)
    
    # Save features and rendered views
    np.save(os.path.join(pc_output_dir, 'fpfh.npy'), features)
    for idx, (view, points) in enumerate(zip(rendered_views, view_points)):
        cv2.imwrite(os.path.join(pc_output_dir, f"view_{idx:03d}.png"), view)
        np.save(os.path.join(pc_output_dir, f'view_{idx:03d}.npy'), points.astype(int))

if __name__ == '__main__':
    # Define dataset parameters directly in code
    DATASET_ROOT = '/datasets/mvtec_3d'  # Original dataset path
    OBJECT_CATEGORY = 'bagel'                # Category to process
    OUTPUT_DIRECTORY = '/datasets/mvtec_3d_multi_view'  # Output directory
    
    # Initialize multi-view renderer with RGB coloring only
    view_renderer = render_utils.MultiViewRender('', color=render_utils.MultiViewRender.COLOR_RGB)
    
    # Process training and test splits
    data_splits = ['train', 'test']
    processed_count = 0
    
    print(f'Processing {OBJECT_CATEGORY}...')
    start_time = time.time()
    
    for split in data_splits:
        split_path = os.path.join(DATASET_ROOT, OBJECT_CATEGORY, split)
        defect_categories = os.listdir(split_path)
        
        for defect in defect_categories:
            # Find all point cloud files
            pc_files = glob.glob(os.path.join(split_path, defect, 'xyz') + "/*.tiff")
            pc_files.sort()
            
            for pc_file in pc_files:
                generate_multiview_data(view_renderer, pc_file, DATASET_ROOT, OUTPUT_DIRECTORY)
                processed_count += 1
                
                if processed_count % 50 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    print(f"Processed {processed_count} files... Time elapsed: {elapsed:.2f} seconds")