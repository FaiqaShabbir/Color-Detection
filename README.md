# Advanced Color Detection Tool

A sophisticated color detection application that identifies colors in images with improved accuracy and user experience.

## Features

### 🎨 **Advanced Color Matching**
- **HSV-based matching**: More accurate color perception using Hue, Saturation, Value
- **Multiple algorithms**: Support for HSV, Euclidean, and Manhattan distance methods
- **Confidence scoring**: Shows how confident the color match is
- **866 named colors**: Comprehensive color database
- **Video processing**: Real-time color detection in video files

### 🖱️ **Enhanced User Interface**
- **Interactive display**: Double-click anywhere to detect colors
- **Real-time feedback**: Instant color information display
- **Method switching**: Press 'h' to cycle through matching algorithms
- **Visual indicators**: Crosshair and confidence indicators
- **Reset functionality**: Press 'r' to clear display

### 🔧 **Technical Improvements**
- **Error handling**: Robust error handling for missing files
- **Modular design**: Clean, maintainable code structure
- **Type hints**: Full type annotation for better development
- **Documentation**: Comprehensive docstrings and comments

## Installation

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage
```bash
python improved_color_detection.py -i your_image.jpg
```

### Examples
```bash
# Use the sample image
python improved_color_detection.py -i colorpic.jpg

# Use a custom color database
python improved_color_detection.py -i image.jpg -d custom_colors.csv

# Process video files
python video_color_detection.py -v your_video.mp4

# Process video with custom database
python video_color_detection.py -v demo.mp4 -d colors.csv
```

### Controls

**For Images (`improved_color_detection.py`):**
- **Double-click**: Detect color at cursor position
- **'h' key**: Cycle through matching methods (HSV → Euclidean → Manhattan)
- **'r' key**: Reset display
- **'ESC' key**: Quit application

**For Videos (`video_color_detection.py`):**
- **Double-click**: Detect color at cursor position
- **'h' key**: Cycle through matching methods (HSV → Euclidean → Manhattan)
- **'SPACE' key**: Pause/resume video
- **'r' key**: Reset color detection
- **'ESC' key**: Quit application

## Color Matching Methods

### 1. HSV (Default - Recommended)
- Uses human color perception model
- Weights hue more heavily than saturation/value
- Most accurate for color identification

### 2. Euclidean Distance
- Standard RGB distance calculation
- Good for general color similarity

### 3. Manhattan Distance
- Original algorithm from basic version
- Simple but less accurate

## File Structure

```
Color-Detection/
├── improved_color_detection.py  # Main application (images)
├── video_color_detection.py     # Video processing application
├── color_detection.py           # Original basic version
├── colors.csv                   # Color database (866 colors)
├── colorpic.jpg                 # Sample image
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Color Database

The `colors.csv` file contains 866 named colors with:
- Color name (e.g., "Air Force Blue (Raf)")
- Hex code (e.g., "#5d8aa8")
- RGB values (0-255 range)

## Technical Details

### Key Improvements Over Original

1. **Better Color Matching**
   - HSV color space for human perception
   - Confidence scoring system
   - Multiple algorithm support

2. **Enhanced UI/UX**
   - Information panel with color details
   - Visual feedback and indicators
   - Keyboard shortcuts for method switching

3. **Code Quality**
   - Object-oriented design
   - Type hints and documentation
   - Error handling and validation
   - Modular architecture

4. **Performance**
   - Preprocessed color database
   - Efficient color matching algorithms
   - Optimized display rendering

## Troubleshooting

### Common Issues

1. **"Image file not found"**
   - Check the image path is correct
   - Ensure the file exists and is readable

2. **"Color database file not found"**
   - Make sure `colors.csv` is in the same directory
   - Or specify the correct path with `-d` option

3. **OpenCV window not displaying**
   - Ensure you have a display environment
   - Try running on a system with GUI support

## Contributing

Feel free to contribute improvements:
- Add new color matching algorithms
- Enhance the user interface
- Add support for different color spaces
- Improve the color database

## License

This project is open source and available under the MIT License. 