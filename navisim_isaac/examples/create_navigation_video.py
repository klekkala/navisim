#!/usr/bin/env python3
"""
Create Video from Navigation Images

This script creates a video/GIF from saved navigation camera images.
"""

import sys
from pathlib import Path
import argparse

# Try to import required libraries
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("ERROR: PIL not available. Install with: pip install Pillow")
    sys.exit(1)

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("Warning: imageio not available. Install with: pip install imageio")


def create_gif(image_dir: Path, output_path: Path, fps: int = 10):
    """
    Create an animated GIF from navigation images.

    Args:
        image_dir: Directory containing PNG images
        output_path: Output GIF file path
        fps: Frames per second
    """
    if not IMAGEIO_AVAILABLE:
        print("ERROR: imageio required to create GIF")
        return

    # Get all PNG images sorted by name
    image_files = sorted(image_dir.glob("step_*.png"))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images")
    print(f"Creating GIF at {fps} FPS...")

    # Load images
    images = []
    for i, img_path in enumerate(image_files):
        if i % 10 == 0:
            print(f"  Loading image {i+1}/{len(image_files)}...")
        images.append(imageio.imread(img_path))

    # Save as GIF
    duration = 1.0 / fps
    imageio.mimsave(output_path, images, duration=duration, loop=0)

    print(f"✓ GIF saved to: {output_path}")
    print(f"  Total frames: {len(images)}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(images)/fps:.1f} seconds")


def create_video(image_dir: Path, output_path: Path, fps: int = 30):
    """
    Create a video (MP4) from navigation images.

    Args:
        image_dir: Directory containing PNG images
        output_path: Output video file path
        fps: Frames per second
    """
    if not IMAGEIO_AVAILABLE:
        print("ERROR: imageio required to create video")
        return

    # Get all PNG images sorted by name
    image_files = sorted(image_dir.glob("step_*.png"))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images")
    print(f"Creating video at {fps} FPS...")

    # Create video writer
    writer = imageio.get_writer(output_path, fps=fps)

    # Write frames
    for i, img_path in enumerate(image_files):
        if i % 10 == 0:
            print(f"  Writing frame {i+1}/{len(image_files)}...")
        frame = imageio.imread(img_path)
        writer.append_data(frame)

    writer.close()

    print(f"✓ Video saved to: {output_path}")
    print(f"  Total frames: {len(image_files)}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(image_files)/fps:.1f} seconds")


def view_images(image_dir: Path):
    """
    Display navigation images with metadata.

    Args:
        image_dir: Directory containing PNG images
    """
    # Get all PNG images sorted by name
    image_files = sorted(image_dir.glob("step_*.png"))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"\nFound {len(image_files)} images in: {image_dir}")
    print("=" * 70)

    for img_path in image_files:
        # Load image
        img = Image.open(img_path)

        # Load metadata if exists
        metadata_path = img_path.with_suffix('.txt')
        metadata = ""
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = f.read().strip()

        print(f"\n{img_path.name}")
        print(f"  Size: {img.size}")
        if metadata:
            for line in metadata.split('\n'):
                print(f"  {line}")


def list_runs(examples_dir: Path):
    """
    List all available navigation runs.

    Args:
        examples_dir: Examples directory containing navigation_output
    """
    output_dir = examples_dir / "navigation_output"

    if not output_dir.exists():
        print(f"No navigation_output directory found in {examples_dir}")
        return

    runs = sorted(output_dir.glob("run_*"))

    if not runs:
        print("No navigation runs found")
        return

    print("\nAvailable navigation runs:")
    print("=" * 70)

    for run_dir in runs:
        image_count = len(list(run_dir.glob("*.png")))
        print(f"\n{run_dir.name}")
        print(f"  Path: {run_dir}")
        print(f"  Images: {image_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Create video/GIF from navigation camera images"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Path to run directory (e.g., navigation_output/run_20241110_120000)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available navigation runs"
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="View images with metadata"
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Create animated GIF"
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Create MP4 video"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for video/GIF (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: navigation.gif or navigation.mp4)"
    )

    args = parser.parse_args()

    examples_dir = Path(__file__).parent

    # List runs
    if args.list:
        list_runs(examples_dir)
        return

    # Get run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # Use latest run
        output_dir = examples_dir / "navigation_output"
        if not output_dir.exists():
            print("No navigation_output directory found")
            print("Run navigation_example.py first to generate images")
            return

        runs = sorted(output_dir.glob("run_*"))
        if not runs:
            print("No navigation runs found")
            print("Run navigation_example.py first to generate images")
            return

        run_dir = runs[-1]  # Use latest run
        print(f"Using latest run: {run_dir.name}")

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return

    # View images
    if args.view:
        view_images(run_dir)
        return

    # Create GIF
    if args.gif:
        output_path = Path(args.output) if args.output else run_dir / "navigation.gif"
        create_gif(run_dir, output_path, fps=args.fps)
        return

    # Create video
    if args.video:
        output_path = Path(args.output) if args.output else run_dir / "navigation.mp4"
        create_video(run_dir, output_path, fps=args.fps)
        return

    # Default: create GIF
    output_path = run_dir / "navigation.gif"
    create_gif(run_dir, output_path, fps=args.fps)


if __name__ == "__main__":
    main()
