import cv2
from collections import defaultdict


def draw_first_fret_boxes(current_scale_notes, current_annotations, guitar_notes, resized_frame, max_x_box, ratio, fret):
    x_min, y_min, x_max, y_max = map(int, max_x_box)  # Convert to integers for drawing
    current_annotations.append({
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max" : y_max,
                    "label": "fret_box"
                })
    box_width = x_max - x_min
    three_fret_box_width = x_min + ratio*(box_width) - 15
    if(three_fret_box_width) < 640:
        return draw_string_lines(current_scale_notes, current_annotations, guitar_notes,resized_frame, x_min, three_fret_box_width, y_min, y_max, fret, ratio)

    
    return resized_frame


def draw_string_lines(current_scale_notes, current_annotations, guitar_notes, resized_frame, x_min, x_max, y_min, y_max, fret, ratio):
    padded_y_min = y_min + 10
    padded_y_max = y_max - 10
    padded_height = padded_y_max - padded_y_min  # Height after padding
    section_height = padded_height // 5  # Divide the remaining height into 5 equal sections

    for i in range(6):
        line_y = padded_y_min + i * section_height
        resized_frame = draw_notes_on_neck(current_scale_notes, current_annotations, guitar_notes, resized_frame, fret, ratio, line_y, x_min, x_max, i)
        cv2.line(resized_frame, 
                 (x_min, line_y), 
                 (x_max, line_y), 
                 (0, 0, 0), 2)  # Blue line with thickness 2
        current_annotations.append({
                    "x_min": x_min,
                    "line_y": line_y,
                    "x_max": x_max,
                    "label": "string"
                })
       
        
    return resized_frame

def get_scale_notes(root: str, chromatic_scale: list, scale_name:str) -> list:
    # Define the major scale pattern: W-W-H-W-W-W-H
    scale_intervals = []
    if scale_name == "Major":
        scale_intervals = [2, 2, 1, 2, 2, 2]
    elif scale_name == "Minor":
        scale_intervals= [2, 1, 2, 2, 1, 2]
    elif scale_name == "Minor Pentatonic":
        scale_intervals =[3, 2, 2, 3, 2]
    elif scale_intervals == "Major Pentatonic":
        scale_intervals = [2, 2, 3, 2, 3]
    
    # Find the root index in the chromatic scale
    if root not in chromatic_scale:
        raise ValueError(f"Root note {root} is not in the chromatic scale.")
    
    root_index = chromatic_scale.index(root)
    major_scale = [root]  # Start with the root note

    # Calculate the major scale using the pattern
    for step in scale_intervals:
        root_index = (root_index + step) % len(chromatic_scale)
        major_scale.append(chromatic_scale[root_index])
    
    return major_scale


def generate_fretboard_notes():
    
    # Define the chromatic scale (12 notes)
    chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Define the open string notes for standard tuning (EADGBE)
    open_strings = ['E', 'A', 'D', 'G', 'B', 'E']

    guitar_notes = defaultdict(lambda: defaultdict(int))
    
    for string_num, open_note in enumerate(open_strings, start=0):
        # Find the starting index of the open note in the chromatic scale
        start_index = chromatic_scale.index(open_note)
        
        # Generate notes for the frets (1-24) and filter for pentatonic notes or set to 0
        for fret in range(1, 25):  # Start from fret 1
            note = chromatic_scale[(start_index + fret) % len(chromatic_scale)]
            guitar_notes[string_num][fret] = note
            
    return guitar_notes
    

def draw_notes_on_neck(current_scale_notes, current_annotations, guitar_notes, resized_frame, fret, ratio, line_y, x_min, x_max, string_num):
    if fret >= 2:
        fret *= 2
    fret_width = (x_max - x_min) // ratio
    
    for i in range(0, ratio):
        note = guitar_notes[string_num ][(fret + i)]
        if note in current_scale_notes:
            cv2.circle(resized_frame,((x_max - (i * fret_width)- fret_width)+ (fret_width//2), line_y), 8 , (0, 255, 255), -1)
            current_annotations.append({
                    "x_min": (x_max - (i * fret_width)- fret_width)+ (fret_width//2),
                    "line_y": line_y,
                    "x_max": x_max,
                    "label": "note"
                })

    return resized_frame
        