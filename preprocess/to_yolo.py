

"""
CrowNNs to_yolo converter script
    Converts csv file into yolo format
"""

import sys
from unittest import result

IMAGE_SIZE = 255

def normalize(x):
    """ Normalize x to [0, 1] rounded to 6 decimal places """
    return round(x / IMAGE_SIZE, 6)

def to_yolo(xmin, ymin, xmax, ymax):
    """ Calculate center, width and height """

    xcenter = (xmin + xmax) / 2
    ycenter = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin

    return normalize(xcenter), normalize(ycenter), normalize(width), normalize(height)

def to_yolo_data(xmin, ymin, xmax, ymax):
    """ Convert to yolo format with hardcoded class 0 """

    xcenter, ycenter, width, height = to_yolo(xmin, ymin, xmax, ymax)
    return '0 {} {} {} {}'.format(xcenter, ycenter, width, height)

def parse_csv(path):
    """ Parse csv file """

    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    lines = [l.split(',') for l in lines]
    return lines

def parse_csv_to_yolo(path, results_path):
    """ Parse csv file to yolo format """
    print('Parsing {}'.format(path))

    lines = parse_csv(path)

    for i, line in enumerate(lines[1:]):
        filename, xmin, ymin, xmax, ymax, label = line
        yolo_line = to_yolo_data(int(xmin), int(ymin), int(xmax), int(ymax))
        """ open file in append mode and write line """
        with open(f'{results_path}/{filename.replace("png", "txt")}', 'a') as f:
            f.write(yolo_line + '\n')    

        """ Display live progress """
        print(f"CURRENT: {i+1} / {len(lines)-1} ({round((i+1)*100 / (len(lines)-1),2)}%)", end="{}".format('\r' if i < len(lines)-2 else '\n'))
        

    print("Parsing complete! Check {}".format(results_path))
    return

if __name__ == "__main__":
    """ Check if arguments are passed """
    if len(sys.argv) < 3:
        print("Usage: py to_yolo.py <csv_file> <results_dir>")
        print("Example: py to_yolo.py data/train.csv data/train_yolo")
        exit(1)

    _, csv_file, results_path = sys.argv

    parse_csv_to_yolo(csv_file, results_path)