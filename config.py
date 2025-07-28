"""
Configuration settings for the Advanced Color Detection Tool.
Modify these values to customize the application behavior.
"""

# Display Settings
DISPLAY_CONFIG = {
    'window_name': 'Advanced Color Detection',
    'font': 'FONT_HERSHEY_SIMPLEX',
    'font_scale': 0.6,
    'thickness': 2,
    'panel_alpha': 0.8,  # Transparency of info panel
    'crosshair_size': 10,
    'crosshair_thickness': 2,
    'color_sample_size': (60, 30),  # Width, Height
}

# Color Matching Settings
MATCHING_CONFIG = {
    'default_method': 'hsv',  # 'hsv', 'euclidean', 'manhattan'
    'hsv_weights': {
        'hue': 0.7,      # Weight for hue difference
        'saturation': 0.2,  # Weight for saturation difference
        'value': 0.1      # Weight for value difference
    },
    'confidence_thresholds': {
        'high': 0.8,     # Green color for confidence display
        'medium': 0.6     # Yellow color for confidence display
    }
}

# UI Colors (BGR format for OpenCV)
UI_COLORS = {
    'background': (0, 0, 0),
    'text_white': (255, 255, 255),
    'text_gray': (128, 128, 128),
    'confidence_high': (0, 255, 0),    # Green
    'confidence_medium': (0, 255, 255), # Yellow
    'confidence_low': (0, 165, 255),    # Orange
    'crosshair': (255, 255, 255),
    'crosshair_outline': (0, 0, 0),
}

# Information Panel Layout
PANEL_CONFIG = {
    'position': (10, 10),
    'size': (400, 120),
    'text_start': (90, 35),
    'line_height': 20,
    'color_sample_pos': (20, 20),
}

# Keyboard Controls
CONTROLS = {
    'quit': 27,           # ESC key
    'method_switch': ord('h'),
    'reset': ord('r'),
    'help': ord('?'),
}

# File Paths
DEFAULT_PATHS = {
    'color_database': 'colors.csv',
    'sample_image': 'colorpic.jpg',
}

# Error Messages
ERROR_MESSAGES = {
    'image_not_found': "Error: Image file '{path}' not found.",
    'image_load_failed': "Error: Could not load image '{path}'.",
    'database_not_found': "Error: Color database file '{path}' not found.",
    'database_load_failed': "Error loading color database: {error}",
}

# Success Messages
SUCCESS_MESSAGES = {
    'image_loaded': "Image loaded successfully: {width}x{height} pixels",
    'database_loaded': "Loaded {count} colors from database",
    'method_switched': "Switched to {method} matching method",
    'display_reset': "Display reset",
}

# Help Text
HELP_TEXT = """
=== Advanced Color Detection ===
Double-click anywhere on the image to detect colors
Press 'h' to cycle through matching methods (HSV/Euclidean/Manhattan)
Press 'r' to reset display
Press '?' to show this help
Press 'ESC' to quit
================================
""" 