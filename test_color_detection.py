#!/usr/bin/env python3
"""
Test script for the Advanced Color Detection Tool.
Tests color matching accuracy and compares different methods.
"""

import cv2
import numpy as np
import pandas as pd
from improved_color_detection import ColorDatabase, ColorInfo
import time


def test_color_matching():
    """Test color matching with different methods."""
    print("Testing Color Matching Methods...")
    print("=" * 50)
    
    # Initialize color database
    color_db = ColorDatabase('colors.csv')
    
    # Test colors (RGB values)
    test_colors = [
        ((255, 0, 0), "Pure Red"),
        ((0, 255, 0), "Pure Green"),
        ((0, 0, 255), "Pure Blue"),
        ((255, 255, 255), "White"),
        ((0, 0, 0), "Black"),
        ((128, 128, 128), "Gray"),
        ((255, 165, 0), "Orange"),
        ((128, 0, 128), "Purple"),
        ((255, 192, 203), "Pink"),
        ((255, 215, 0), "Gold"),
    ]
    
    methods = ['hsv', 'euclidean', 'manhattan']
    
    for rgb, color_name in test_colors:
        print(f"\nTesting {color_name} (RGB: {rgb}):")
        print("-" * 40)
        
        for method in methods:
            start_time = time.time()
            result = color_db.find_closest_color(rgb, method)
            end_time = time.time()
            
            print(f"{method.upper():12} -> {result.name:25} "
                  f"(Confidence: {result.confidence:.3f}, "
                  f"Time: {(end_time - start_time)*1000:.2f}ms)")


def test_performance():
    """Test performance with random colors."""
    print("\n\nPerformance Test with Random Colors...")
    print("=" * 50)
    
    color_db = ColorDatabase('colors.csv')
    methods = ['hsv', 'euclidean', 'manhattan']
    
    # Generate 100 random colors
    np.random.seed(42)  # For reproducible results
    random_colors = np.random.randint(0, 256, (100, 3))
    
    for method in methods:
        print(f"\nTesting {method.upper()} method:")
        start_time = time.time()
        
        for rgb in random_colors:
            color_db.find_closest_color(tuple(rgb), method)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(random_colors) * 1000  # Convert to milliseconds
        
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time per color: {avg_time:.2f}ms")


def test_accuracy_comparison():
    """Compare accuracy of different methods using known colors."""
    print("\n\nAccuracy Comparison Test...")
    print("=" * 50)
    
    color_db = ColorDatabase('colors.csv')
    
    # Known color mappings (RGB -> expected color name)
    known_colors = {
        (255, 0, 0): "Red",
        (0, 255, 0): "Green", 
        (0, 0, 255): "Blue",
        (255, 255, 0): "Yellow",
        (255, 0, 255): "Magenta",
        (0, 255, 255): "Cyan",
        (255, 255, 255): "White",
        (0, 0, 0): "Black",
    }
    
    methods = ['hsv', 'euclidean', 'manhattan']
    
    for method in methods:
        print(f"\n{method.upper()} Method Results:")
        print("-" * 30)
        
        correct_matches = 0
        total_colors = len(known_colors)
        
        for rgb, expected in known_colors.items():
            result = color_db.find_closest_color(rgb, method)
            is_correct = expected.lower() in result.name.lower()
            
            if is_correct:
                correct_matches += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"  {status} Expected: {expected:10} -> Got: {result.name}")
        
        accuracy = (correct_matches / total_colors) * 100
        print(f"  Accuracy: {accuracy:.1f}% ({correct_matches}/{total_colors})")


def test_color_database_integrity():
    """Test the color database for integrity and completeness."""
    print("\n\nColor Database Integrity Test...")
    print("=" * 50)
    
    try:
        color_db = ColorDatabase('colors.csv')
        df = color_db.df
        
        print(f"✓ Database loaded successfully")
        print(f"  Total colors: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() == 0:
            print("✓ No missing values found")
        else:
            print("✗ Missing values found:")
            print(missing_values[missing_values > 0])
        
        # Check RGB value ranges
        rgb_columns = ['R', 'G', 'B']
        for col in rgb_columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if 0 <= min_val <= max_val <= 255:
                print(f"✓ {col} values in valid range: {min_val}-{max_val}")
            else:
                print(f"✗ {col} values out of range: {min_val}-{max_val}")
        
        # Check for duplicate colors
        duplicates = df.duplicated(subset=['R', 'G', 'B']).sum()
        if duplicates == 0:
            print("✓ No duplicate RGB values found")
        else:
            print(f"✗ {duplicates} duplicate RGB values found")
        
        # Sample some colors
        print("\nSample colors from database:")
        sample_colors = df.sample(min(5, len(df)))
        for _, row in sample_colors.iterrows():
            print(f"  {row['color_name']:20} - RGB({row['R']:3}, {row['G']:3}, {row['B']:3})")
            
    except Exception as e:
        print(f"✗ Database integrity test failed: {e}")


def main():
    """Run all tests."""
    print("Advanced Color Detection - Test Suite")
    print("=" * 60)
    
    try:
        # Run tests
        test_color_database_integrity()
        test_color_matching()
        test_performance()
        test_accuracy_comparison()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        
    except Exception as e:
        print(f"\nTest suite failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 