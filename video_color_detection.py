#!/usr/bin/env python3
"""
Video Color Detection Application
Analyzes video files frame by frame to detect colors in real-time.
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import sys
import os
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import colorsys
from pathlib import Path
import time


@dataclass
class ColorInfo:
    """Data class to store color information."""
    name: str
    hex_code: str
    rgb: Tuple[int, int, int]
    hsv: Tuple[float, float, float]
    confidence: float


class ColorDatabase:
    """Manages the color database and provides color matching functionality."""
    
    def __init__(self, csv_path: str = 'colors.csv'):
        """Initialize the color database from CSV file."""
        try:
            self.df = pd.read_csv(csv_path, names=["color", "color_name", "hex", "R", "G", "B"], header=None)
            self._preprocess_colors()
        except FileNotFoundError:
            print(f"Error: Color database file '{csv_path}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading color database: {e}")
            sys.exit(1)
    
    def _preprocess_colors(self):
        """Preprocess colors for better matching."""
        # Convert RGB to HSV for better color matching
        self.df['HSV'] = self.df.apply(
            lambda row: colorsys.rgb_to_hsv(row['R']/255, row['G']/255, row['B']/255), axis=1
        )
    
    def find_closest_color(self, rgb: Tuple[int, int, int], method: str = 'hsv') -> ColorInfo:
        """
        Find the closest color using specified matching method.
        
        Args:
            rgb: RGB values (0-255)
            method: Matching method ('hsv', 'euclidean', 'manhattan')
        
        Returns:
            ColorInfo object with best match
        """
        r, g, b = rgb
        
        if method == 'hsv':
            # Convert to HSV for better color perception
            hsv = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            min_distance = float('inf')
            best_match = None
            
            for idx, row in self.df.iterrows():
                # Calculate HSV distance (weighted for human perception)
                h_diff = min(abs(hsv[0] - row['HSV'][0]), 1 - abs(hsv[0] - row['HSV'][0]))
                s_diff = abs(hsv[1] - row['HSV'][1])
                v_diff = abs(hsv[2] - row['HSV'][2])
                
                # Weight hue more heavily as it's most important for color perception
                distance = h_diff * 0.7 + s_diff * 0.2 + v_diff * 0.1
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = row
            
        elif method == 'euclidean':
            # Euclidean distance in RGB space
            distances = np.sqrt(
                (self.df['R'] - r)**2 + 
                (self.df['G'] - g)**2 + 
                (self.df['B'] - b)**2
            )
            best_idx = distances.idxmin()
            best_match = self.df.iloc[best_idx]
            min_distance = distances[best_idx]
            
        else:  # manhattan
            # Manhattan distance (original method)
            distances = abs(self.df['R'] - r) + abs(self.df['G'] - g) + abs(self.df['B'] - b)
            best_idx = distances.idxmin()
            best_match = self.df.iloc[best_idx]
            min_distance = distances[best_idx]
        
        # Calculate confidence (0-1, higher is better)
        max_possible_distance = 441.67 if method == 'euclidean' else 765 if method == 'manhattan' else 1.0
        confidence = 1 - (min_distance / max_possible_distance)
        
        return ColorInfo(
            name=best_match['color_name'],
            hex_code=best_match['hex'],
            rgb=(best_match['R'], best_match['G'], best_match['B']),
            hsv=best_match['HSV'],
            confidence=confidence
        )


class VideoColorDetector:
    """Video color detection application."""
    
    def __init__(self, video_path: str, color_db: ColorDatabase):
        """Initialize the video color detector."""
        self.video_path = video_path
        self.color_db = color_db
        self.method = 'hsv'  # Default to HSV matching
        self.paused = False
        self.current_frame = None
        self.current_color = None
        self.clicked = False
        self.xpos = self.ypos = 0
        
        # Load video
        self.cap = self._load_video()
        if self.cap is None:
            sys.exit(1)
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video loaded: {self.width}x{self.height} @ {self.fps:.2f} FPS")
        print(f"Total frames: {self.frame_count}")
        
        # Create window and set up mouse callback
        cv2.namedWindow('Video Color Detection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Video Color Detection', self._mouse_callback)
        
        # Display settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
    def _load_video(self) -> Optional[cv2.VideoCapture]:
        """Load and validate video file."""
        if not os.path.exists(self.video_path):
            print(f"Error: Video file '{self.video_path}' not found.")
            return None
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{self.video_path}'.")
            return None
        
        return cap
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDBLCLK and self.current_frame is not None:
            self.clicked = True
            self.xpos, self.ypos = x, y
            
            # Get BGR values (OpenCV uses BGR)
            b, g, r = self.current_frame[y, x]
            
            # Find closest color
            self.current_color = self.color_db.find_closest_color((r, g, b), self.method)
            
            print(f"Frame {self.cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} - "
                  f"Clicked at ({x}, {y}) - RGB: ({r}, {g}, {b})")
            print(f"Detected color: {self.current_color.name} "
                  f"(Confidence: {self.current_color.confidence:.2f})")
    
    def _draw_info_panel(self, frame: np.ndarray, frame_num: int):
        """Draw information panel with video and color details."""
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Draw background rectangle
        cv2.rectangle(overlay, (10, 10), (450, 140), (0, 0, 0), -1)
        
        # Add transparency
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw color sample if available
        if self.current_color:
            cv2.rectangle(frame, (20, 20), (80, 50), self.current_color.rgb, -1)
            cv2.rectangle(frame, (20, 20), (80, 50), (255, 255, 255), 2)
        
        # Draw text information
        y_offset = 35
        line_height = 20
        
        # Video info
        cv2.putText(frame, f"Frame: {frame_num}/{self.frame_count}", 
                   (90, y_offset), self.font, 0.5, (255, 255, 255), 1)
        
        # Color info
        if self.current_color:
            cv2.putText(frame, f"Color: {self.current_color.name}", 
                       (90, y_offset + line_height), self.font, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"RGB: ({self.current_color.rgb[0]}, {self.current_color.rgb[1]}, {self.current_color.rgb[2]})", 
                       (90, y_offset + 2*line_height), self.font, 0.4, (255, 255, 255), 1)
            
            confidence_color = (0, 255, 0) if self.current_color.confidence > 0.8 else (0, 255, 255) if self.current_color.confidence > 0.6 else (0, 165, 255)
            cv2.putText(frame, f"Confidence: {self.current_color.confidence:.2f}", 
                       (90, y_offset + 3*line_height), self.font, 0.4, confidence_color, 1)
        
        # Method and status
        cv2.putText(frame, f"Method: {self.method.upper()}", 
                   (90, y_offset + 4*line_height), self.font, 0.4, (128, 128, 128), 1)
        
        # Status
        status = "PAUSED" if self.paused else "PLAYING"
        status_color = (0, 165, 255) if self.paused else (0, 255, 0)
        cv2.putText(frame, f"Status: {status}", 
                   (90, y_offset + 5*line_height), self.font, 0.4, status_color, 1)
    
    def _draw_crosshair(self, frame: np.ndarray):
        """Draw crosshair at clicked position."""
        if self.clicked and self.current_color:
            # Draw crosshair
            cv2.line(frame, (self.xpos - 10, self.ypos), (self.xpos + 10, self.ypos), (255, 255, 255), 2)
            cv2.line(frame, (self.xpos, self.ypos - 10), (self.xpos, self.ypos + 10), (255, 255, 255), 2)
            cv2.circle(frame, (self.xpos, self.ypos), 15, (0, 0, 0), 2)
    
    def run(self):
        """Main video processing loop."""
        print("\n=== Video Color Detection ===")
        print("Double-click anywhere on the video to detect colors")
        print("Press 'h' to cycle through matching methods (HSV/Euclidean/Manhattan)")
        print("Press 'SPACE' to pause/resume video")
        print("Press 'r' to reset color detection")
        print("Press 'ESC' to quit")
        print("=" * 35)
        
        frame_delay = int(1000 / self.fps)  # Delay in milliseconds
        
        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    # Video ended, restart
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        break
            
            if frame is not None:
                self.current_frame = frame.copy()
                frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                # Create a copy for drawing
                display_frame = frame.copy()
                
                # Draw UI elements
                self._draw_crosshair(display_frame)
                self._draw_info_panel(display_frame, frame_num)
                
                # Show frame
                cv2.imshow('Video Color Detection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(frame_delay) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('h'):  # Cycle through methods
                methods = ['hsv', 'euclidean', 'manhattan']
                current_idx = methods.index(self.method)
                self.method = methods[(current_idx + 1) % len(methods)]
                print(f"Switched to {self.method.upper()} matching method")
            elif key == ord(' '):  # SPACE - pause/resume
                self.paused = not self.paused
                print(f"Video {'paused' if self.paused else 'resumed'}")
            elif key == ord('r'):  # Reset
                self.clicked = False
                self.current_color = None
                print("Color detection reset")
        
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function to run the video color detection application."""
    parser = argparse.ArgumentParser(
        description="Video Color Detection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_color_detection.py -v your_video.mp4
  python video_color_detection.py -v demo.mp4 -d colors.csv
        """
    )
    
    parser.add_argument('-v', '--video', required=True, 
                       help="Path to the video file")
    parser.add_argument('-d', '--database', default='colors.csv',
                       help="Path to color database CSV file (default: colors.csv)")
    
    args = parser.parse_args()
    
    # Initialize color database
    print("Loading color database...")
    color_db = ColorDatabase(args.database)
    print(f"Loaded {len(color_db.df)} colors from database")
    
    # Initialize and run video color detector
    detector = VideoColorDetector(args.video, color_db)
    detector.run()


if __name__ == "__main__":
    main() 