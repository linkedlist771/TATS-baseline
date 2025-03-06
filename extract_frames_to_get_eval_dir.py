import cv2
import os
import numpy as np
import math
import argparse

class AdvancedFrameExtractor:
    def __init__(self, output_dir="extracted_frames"):
        """Initialize the FrameExtractor with an output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def extract_frames(self, video_path, every_n_frames=1, max_frames=None):
        """Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            every_n_frames: Extract every n-th frame
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of extracted frames as numpy arrays
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise Exception(f"Error opening video file: {video_path}")
            
        frames = []
        count = 0
        frame_count = 0
        
        while True:
            success, frame = video.read()
            if not success:
                break
                
            if count % every_n_frames == 0:
                frames.append(frame)
                frame_count += 1
                
                if max_frames is not None and frame_count >= max_frames:
                    break
                    
            count += 1
            
        video.release()
        return frames
    
    def resize_frame(self, frame, resolution):
        """Resize frame to the target resolution maintaining aspect ratio.
        
        Args:
            frame: Input frame
            resolution: Target resolution (will be used for the smaller dimension)
            
        Returns:
            Resized frame
        """
        h, w = frame.shape[:2]
        
        # Calculate scale based on the smaller dimension
        scale = resolution / min(h, w)
        
        # Calculate target size based on aspect ratio
        if h < w:
            target_size = (int(w * scale), resolution)
        else:
            target_size = (resolution, int(h * scale))
            
        # Resize the image
        resized = cv2.resize(frame, target_size)
        
        # Center crop if needed
        h, w = resized.shape[:2]
        if h > resolution or w > resolution:
            h_start = (h - resolution) // 2
            w_start = (w - resolution) // 2
            resized = resized[h_start:h_start+resolution, w_start:w_start+resolution]
            
        return resized
    
    def save_frames(self, frames, output_subdir, start_index=1, resolution=None):
        """Save extracted frames to the output directory.
        
        Args:
            frames: List of frames to save
            output_subdir: Subdirectory within output_dir to save frames
            start_index: Starting index for frame numbering
            resolution: Target resolution for resizing frames
            
        Returns:
            List of saved file paths and next index to use
        """
        full_output_dir = os.path.join(self.output_dir, output_subdir)
        os.makedirs(full_output_dir, exist_ok=True)
        
        saved_paths = []
        current_index = start_index
        
        for frame in frames:
            if resolution is not None:
                frame = self.resize_frame(frame, resolution)
                
            output_path = os.path.join(full_output_dir, f"{current_index}.png")
            cv2.imwrite(output_path, frame)
            saved_paths.append(output_path)
            current_index += 1
            
        return saved_paths, current_index
    
    def process_video_directory(self, video_dir, output_subdir, resolution=None, every_n_frames=1, max_frames=None):
        """Process all videos in a directory and save frames.
        
        Args:
            video_dir: Directory containing videos
            output_subdir: Subdirectory within output_dir to save frames
            resolution: Target resolution for resizing frames
            every_n_frames: Extract every n-th frame
            max_frames: Maximum number of frames to extract per video
            
        Returns:
            List of all saved file paths
        """
        if not os.path.exists(video_dir):
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
            
        all_saved_paths = []
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        next_index = 1  # Start numbering from 1
        
        for filename in os.listdir(video_dir):
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(video_dir, filename)
                
                try:
                    frames = self.extract_frames(video_path, every_n_frames, max_frames)
                    saved_paths, next_index = self.save_frames(
                        frames, 
                        output_subdir, 
                        start_index=next_index,
                        resolution=resolution
                    )
                    all_saved_paths.extend(saved_paths)
                    print(f"Processed {video_path}: extracted {len(frames)} frames")
                except Exception as e:
                    print(f"Error processing {video_path}: {str(e)}")
                    
        return all_saved_paths
    
    def process_directories(self, true_video_dir, generated_video_dir, resolution=None, every_n_frames=1, max_frames=None):
        """Process videos from two directories and save frames.
        
        Args:
            true_video_dir: Directory containing true videos
            generated_video_dir: Directory containing generated videos
            resolution: Target resolution for resizing frames
            every_n_frames: Extract every n-th frame
            max_frames: Maximum number of frames to extract per video
            
        Returns:
            Tuple of (true_paths, generated_paths) lists of frame paths
        """
        print(f"Processing true videos from {true_video_dir}...")
        true_paths = self.process_video_directory(true_video_dir, "true", resolution, every_n_frames, max_frames)
        
        print(f"Processing generated videos from {generated_video_dir}...")
        generated_paths = self.process_video_directory(generated_video_dir, "generated", resolution, every_n_frames, max_frames)
        
        return true_paths, generated_paths


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract frames from video directories.')
    parser.add_argument('--true_video_dir', type=str, required=True,
                        help='Directory containing true/original videos')
    parser.add_argument('--generated_video_dir', type=str, required=True,
                        help='Directory containing generated videos')
    parser.add_argument('--output_dir', type=str, default='extracted_frames',
                        help='Output directory for extracted frames')
    parser.add_argument('--resolution', type=int, default=128,
                        help='Target resolution for resizing frames')
    parser.add_argument('--every_n_frames', type=int, default=1,
                        help='Extract every n-th frame')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to extract per video')
    
    args = parser.parse_args()
    
    # Create the frame extractor
    extractor = AdvancedFrameExtractor(output_dir=args.output_dir)
    
    # Process both directories
    true_paths, generated_paths = extractor.process_directories(
        true_video_dir=args.true_video_dir,
        generated_video_dir=args.generated_video_dir,
        resolution=args.resolution,
        every_n_frames=args.every_n_frames,
        max_frames=args.max_frames
    )
    
    # Print summary
    print(f"\nExtraction complete!")
    print(f"True frames extracted: {len(true_paths)}")
    print(f"Generated frames extracted: {len(generated_paths)}")
    print(f"Frames saved to {args.output_dir}")