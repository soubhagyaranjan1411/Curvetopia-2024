from flask import Flask, request, jsonify, send_file, render_template
import os
import io
import zipfile
import numpy as np
from svgpathtools import svg2paths
import cv2
from scipy.optimize import leastsq
from scipy.interpolate import UnivariateSpline
from shapely.geometry import Polygon
import csv

app = Flask(__name__)


# Your existing functions go here...

# 1. Load SVG File
def load_svg(file_path):
    paths, attributes = svg2paths(file_path)
    return paths, attributes


# 2. Extract Points from Path
def extract_points_from_path(path):
    points = []
    for segment in path:
        if segment.__class__.__name__ == 'Line':
            points.append((segment.start.real, segment.start.imag))
            points.append((segment.end.real, segment.end.imag))
        elif segment.__class__.__name__ == 'CubicBezier':
            points.append((segment.start.real, segment.start.imag))
            points.append((segment.control1.real, segment.control1.imag))
            points.append((segment.control2.real, segment.control2.imag))
            points.append((segment.end.real, segment.end.imag))
    return np.array(points, dtype=np.float32)


# 3. Detect Shape Type
def detect_shape(points):
    if len(points) < 5:
        return 'complex'

    points = np.array(points, dtype=np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)
    width, height = rect[1]
    aspect_ratio = max(width, height) / min(width, height)

    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    radius_std = np.std(distances)

    if radius_std < 0.1 * np.mean(distances):
        return 'circle'

    if aspect_ratio < 1.2:
        return 'rectangle'

    hull = cv2.convexHull(points)
    hull_vertices = hull.shape[0]
    if hull_vertices > 5:
        return 'star'

    if len(points) >= 3:
        polygon = Polygon(points)
        if polygon.is_valid and len(polygon.exterior.coords) == 3:
            return 'triangle'

    return 'complex'


# 4. Regularize Shapes
def fit_circle(points):
    def objective_function(params, points):
        cx, cy, r = params
        distances = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        return distances - r

    cx0, cy0 = np.mean(points, axis=0)
    r0 = np.mean(np.sqrt((points[:, 0] - cx0) ** 2 + (points[:, 1] - cy0) ** 2))
    initial_guess = [cx0, cy0, r0]

    result, _ = leastsq(objective_function, initial_guess, args=(points,))
    cx, cy, r = result

    return {
        'type': 'circle',
        'center': (cx, cy),
        'radius': r
    }


def fit_rectangle(points):
    if len(points) < 4:
        print("Not enough points for rectangle fitting")
        return {'type': 'rectangle', 'box': []}

    points = np.array(points, dtype=np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)
    return {'type': 'rectangle', 'box': box}


def fit_star(points):
    hull = cv2.convexHull(points)
    hull = np.array(hull, dtype=np.float32)
    if hull.ndim == 3:
        hull = hull[:, 0, :]
    return {
        'type': 'star',
        'vertices': hull
    }


def fit_triangle(points):
    hull = cv2.convexHull(points)
    if len(hull) >= 3:
        triangle = hull[:3]
    else:
        triangle = hull
    return {
        'type': 'triangle',
        'vertices': triangle
    }


def fit_spline(points):
    points = np.array(points)
    if len(points) < 4:
        print("Not enough points for spline fitting")
        return {'type': 'complex', 'points': points}

    x = points[:, 0]
    y = points[:, 1]

    # Sort points based on x values
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    # Handle duplicate x values by averaging corresponding y values
    unique_x, unique_indices = np.unique(x, return_index=True)
    y = [np.mean(y[np.where(x == xi)]) for xi in unique_x]
    x = unique_x

    # Fit the spline
    spline = UnivariateSpline(x, y, s=0, k=3)
    x_new = np.linspace(x.min(), x.max(), 100)
    y_new = spline(x_new)

    regularized_points = np.vstack((x_new, y_new)).T
    return {
        'type': 'complex',
        'points': regularized_points
    }


def regularize_shape(points, shape_type):
    if shape_type == 'circle':
        return fit_circle(points)
    elif shape_type == 'rectangle':
        return fit_rectangle(points)
    elif shape_type == 'star':
        return fit_star(points)
    elif shape_type == 'triangle':
        return fit_triangle(points)
    else:
        return fit_spline(points)


# 5. Detect Symmetry
def detect_symmetry(points):
    def find_symmetric_pairs(points, line):
        symmetric_pairs = []
        for point in points:
            reflected_point = reflect_point(point, line)
            if tuple(reflected_point) in map(tuple, points):
                symmetric_pairs.append((point, reflected_point))
        return symmetric_pairs

    def reflect_point(point, line):
        x0, y0 = point
        x1, y1 = line[0]
        x2, y2 = line[1]
        A = x2 - x1
        B = y2 - y1
        C = x1 * y2 - x2 * y1
        D = A * A - B * B
        E = 2 * (A * B)
        x = ((B * B - A * A) * x0 + E * y0 - 2 * A * C) / D
        y = ((A * A - B * B) * y0 + E * x0 + 2 * B * C) / D
        return (x, y)

    x_coords = points[:, 0]
    y_coords = points[:, 1]
    x_mean = np.mean(x_coords)
    y_mean = np.mean(y_coords)

    vertical_line = [(x_mean, min(y_coords)), (x_mean, max(y_coords))]
    horizontal_line = [(min(x_coords), y_mean), (max(x_coords), y_mean)]

    vertical_symmetry = find_symmetric_pairs(points, vertical_line)
    horizontal_symmetry = find_symmetric_pairs(points, horizontal_line)

    print(f"Vertical symmetry line: {vertical_line}")
    print(f"Horizontal symmetry line: {horizontal_line}")

    return vertical_symmetry, horizontal_symmetry


# 6. Convert Regularized Shapes to SVG Paths
def convert_shape_to_svg_path(shape):
    if 'type' not in shape:
        print(f"Invalid shape format: {shape}")
        return ""

    if shape['type'] == 'circle':
        if 'center' in shape and 'radius' in shape:
            cx, cy = shape['center']
            r = shape['radius']
            svg_path = f"M {cx - r},{cy} A {r},{r} 0 1,0 {cx + r},{cy} A {r},{r} 0 1,0 {cx - r},{cy} Z"
        else:
            print(f"Missing circle parameters: {shape}")
            svg_path = ""
    elif shape['type'] == 'rectangle':
        if 'box' in shape:
            box = shape['box']
            svg_path = "M " + " L ".join(f"{x},{y}" for x, y in box) + " Z"
        else:
            print(f"Missing rectangle box: {shape}")
            svg_path = ""
    elif shape['type'] == 'star':
        if 'vertices' in shape:
            vertices = shape['vertices']
            if vertices.ndim == 2 and vertices.shape[1] == 2:
                svg_path = "M " + " L ".join(f"{x},{y}" for x, y in vertices) + " Z"
            else:
                print(f"Unexpected star vertices format: {vertices}")
                svg_path = ""
        else:
            print(f"Missing star vertices: {shape}")
            svg_path = ""
    elif shape['type'] == 'triangle':
        if 'vertices' in shape:
            vertices = shape['vertices']
            if vertices.ndim == 2 and vertices.shape[1] == 2:
                svg_path = "M " + " L ".join(f"{x},{y}" for x, y in vertices) + " Z"
            else:
                print(f"Unexpected triangle vertices format: {vertices}")
                svg_path = ""
        else:
            print(f"Missing triangle vertices: {shape}")
            svg_path = ""
    elif shape['type'] == 'complex':
        if 'points' in shape:
            points = shape['points']
            svg_path = "M " + " L ".join(f"{x},{y}" for x, y in points) + " Z"
        else:
            print(f"Missing complex shape points: {shape}")
            svg_path = ""
    else:
        print(f"Unknown shape type: {shape['type']}")
        svg_path = ""

    return svg_path


# 7. Save SVG File
def save_svg(file_path, shapes):
    svg_header = '<?xml version="1.0" encoding="UTF-8"?>\n<svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">\n'
    svg_footer = '</svg>'
    svg_content = svg_header

    for shape in shapes:
        svg_path = convert_shape_to_svg_path(shape)
        if svg_path:
            svg_content += f'<path d="{svg_path}" fill="none" stroke="black" />\n'

    svg_content += svg_footer

    with open(file_path, 'w') as file:
        file.write(svg_content)


# 8. Convert Shapes to CSV Format
def convert_shape_to_csv(shape, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if shape['type'] == 'circle':
            cx, cy = shape['center']
            r = shape['radius']
            csvwriter.writerow(['type', 'cx', 'cy', 'r'])
            csvwriter.writerow(['circle', cx, cy, r])
        elif shape['type'] == 'rectangle':
            csvwriter.writerow(['type', 'x', 'y'])
            csvwriter.writerow(['rectangle'])
            for x, y in shape['box']:
                csvwriter.writerow([x, y])
        elif shape['type'] == 'star':
            csvwriter.writerow(['type', 'x', 'y'])
            csvwriter.writerow(['star'])
            for x, y in shape['vertices']:
                csvwriter.writerow([x, y])
        elif shape['type'] == 'triangle':
            csvwriter.writerow(['type', 'x', 'y'])
            csvwriter.writerow(['triangle'])
            for x, y in shape['vertices']:
                csvwriter.writerow([x, y])
        elif shape['type'] == 'complex':
            csvwriter.writerow(['type', 'x', 'y'])
            csvwriter.writerow(['complex'])
            for x, y in shape['points']:
                csvwriter.writerow([x, y])


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = 'uploaded.svg'
        file.save(file_path)
        paths, attributes = load_svg(file_path)
        shapes = []
        for path in paths:
            points = extract_points_from_path(path)
            shape_type = detect_shape(points)
            regularized_shape = regularize_shape(points, shape_type)
            shapes.append(regularized_shape)

        # Save the SVG and CSV files
        output_svg_path = 'output.svg'
        save_svg(output_svg_path, shapes)

        # Create a zip file in memory
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            zf.writestr('output.svg', open(output_svg_path, 'rb').read())

            # Save each shape as a separate CSV file
            for i, shape in enumerate(shapes):
                csv_filename = f'shape_{i}.csv'
                csv_file_path = f'shape_{i}.csv'
                convert_shape_to_csv(shape, csv_file_path)
                zf.writestr(csv_filename, open(csv_file_path, 'rb').read())
                os.remove(csv_file_path)  # Clean up the CSV file after adding to the ZIP

        memory_file.seek(0)

        # Remove temporary files
        os.remove(file_path)
        os.remove(output_svg_path)

        return send_file(memory_file, download_name='result.zip', as_attachment=True)



