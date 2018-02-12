from __future__ import print_function
import subprocess
import os
import shutil
import thread
import time
from scipy import misc
import matplotlib.pyplot as plt
from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw, ImageFont

path_to_images = "./Images_crrt/"
name_image = "image_in.jpg"


def setup_image_folder(path_to_images):
    """give a clean folder in which to put the images taken by the webcam and the
    processed results."""

    print("setup images folder...")

    if os.path.isdir(path_to_images):
        print("folder already exists: remove...")
        shutil.rmtree(path_to_images)

    os.mkdir(path_to_images)
    print("folder created")


def input_thread(L):
    """a separate thread for waiting for keyboard input while taking shots with
    the webcam."""
    raw_input()
    L.append(None)


def get_image(path_to_images, name_image):
    """take shots with the webcam untill stopped by the user pressing a key. Save
    the images in the image folder."""

    print("press enter when happy with image\r", end="")
    L = []
    thread.start_new_thread(input_thread, (L,))
    plt.ion()
    image_number = 0
    while 1:
        if image_number == 0:
            let_camera_update_parameters(path_to_images, name_image)
        else:
            take_one_shot(path_to_images, name_image)
        show_shot(path_to_images, name_image)
        time.sleep(0.5)

        image_number += 1

        if L:
            break

        os.remove("./{}/{}".format(path_to_images, name_image))

    plt.close()


def take_one_shot(path_to_images, name_image, video_source="/dev/video0"):
    """take one shot using the webcamera by default"""
    subprocess_cmd("ffmpeg -f video4linux2 -s 1280x720 -i {} -frames 1 ./{}/{} -loglevel error -nostats".format(video_source, path_to_images, name_image))


def let_camera_update_parameters(path_to_images, name_image, video_source="/dev/video0"):
    """take one shot using the webcamera by default, giving a bit of time to the came to update its parameters"""
    subprocess_cmd("ffmpeg -f video4linux2 -s 1280x720 -i {} -ss 00:00:02 -frames 1 ./{}/{} -loglevel error -nostats".format(video_source, path_to_images, name_image))


def show_shot(path_to_images, name_image):
    """show the shot that has just been taken."""
    crrt_image = misc.imread("./{}/{}".format(path_to_images, name_image))

    plt.imshow(crrt_image)

    plt.draw()
    plt.pause(0.5)


def subprocess_cmd(command):
    """execute a bash command and return output"""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()

    return(proc_stdout)


def detect_face(face_file, max_results=10):
    """From Google Tutorial.

    Uses the Vision API to detect faces in the given file.
    Args:
        face_file: A file-like object containing an image with faces.
    Returns:
        An array of Face objects with information about the picture.
    """
    client = vision.ImageAnnotatorClient()

    content = face_file.read()
    image = types.Image(content=content)

    return client.face_detection(image=image).face_annotations


def likelihood(likelihood_value):
    return {
        1: "VERY UNLIKELY",
        2: "UNLIKELY",
        3: "POSSIBLE",
        4: "LIKELY",
        5: "VERY LIKELY"
    }.get(likelihood_value, 9)


def generate_string_label(list_emotions, list_emotion_scores):
    max_value = max(list_emotion_scores)
    max_likelihood = likelihood(max_value)

    if max_value == 1:
        return("Unemotional")

    else:
        list_most_likely = [i for i, j in enumerate(list_emotion_scores) if j == max_value]
        list_strings = [max_likelihood]

        for crrt_most_likely in list_most_likely:
            list_strings.append(list_emotions[crrt_most_likely])

        string_label = " ".join(list_strings)

    return(string_label)


def draw_line_list_points(draw_object, list_points, list_point_coords, close=True):
    """draw lines in image.
    """
    if close:
        list_points.append(list_points[0])

    width_line = 4

    point_start = list_points[0: -1]
    point_end = list_points[1:]

    for crrt_start, crrt_end in zip(point_start, point_end):
        x1 = list_point_coords[crrt_start][0]
        y1 = list_point_coords[crrt_start][1]
        x2 = list_point_coords[crrt_end][0]
        y2 = list_point_coords[crrt_end][1]
        draw_object.line((x1, y1, x2, y2), width=width_line, fill="blue")


def highlight_faces(image, faces, output_filename, terminal_print=True):
    """Adapted from Google tutorial. Draw figure with API information and save it.

    Note: this is just for illustration, the graphics are not robust: hard coded
    fonts etc.
    """
    im = Image.open(image)
    draw = ImageDraw.Draw(im)

    for (face_ind, face) in enumerate(faces):

        # compute emotions
        list_emotion_scores = [face.sorrow_likelihood,
                               face.joy_likelihood,
                               face.anger_likelihood,
                               face.surprise_likelihood]

        list_emotions = ["SORROW",
                         "JOY",
                         "ANGER",
                         "SURPRISE"]

        string_label = generate_string_label(list_emotions, list_emotion_scores)

        if terminal_print:
            # print emotions on terminal
            print("\n")
            print("-----------------------")
            print("Face {}".format(face_ind))

            for (crrt_emotion, crrt_score) in zip(list_emotions, list_emotion_scores):
                print("{}: {}".format(crrt_emotion, crrt_score))

            print(string_label)

            print("-----------------------")

        # draw box around face
        box = [(vertex.x, vertex.y)
               for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')

        # add legend in the face box
        fontsize = 35
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", fontsize)

        offset = 5
        heigth_text = 40
        length_text = box[1][0] - box[0][0] - 2 * offset
        draw.rectangle(((box[0][0] + offset, box[0][1] + offset), (box[0][0] + length_text + offset, box[0][1] + heigth_text + offset)), fill="black")
        draw.text((box[0][0] + offset, box[0][1] + offset), string_label, font=font, fill=(255, 255, 255, 255))

        # highlight significant points
        point_nbr = 0
        half_width_sqare = 2

        list_point_coords = []

        for point in face.landmarks:
            x = point.position.x
            y = point.position.y

            list_point_coords.append((x, y))

            draw.rectangle(((x - half_width_sqare, y - half_width_sqare), (x + half_width_sqare, y + half_width_sqare)), fill="red")

            # fontsize = 15
            # font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", fontsize)
            # draw.text((x, y), str(point_nbr), font=font, fill=(255, 255, 0, 0))

            point_nbr += 1

        all_lists_points = [
                           [10, 11, 9],
                           [10, 12, 11],
                           [14, 7, 13, 15],
                           [7, 6],
                           [14, 6, 13, 7, 14],
                           [16, 17, 18, 19],
                           [21, 22, 23, 24],
                           [30, 6],
        ]

        for crrt_list_points in all_lists_points:
            draw_line_list_points(draw, crrt_list_points, list_point_coords)

        draw_line_list_points(draw, [2, 26, 3], list_point_coords, close=False)
        draw_line_list_points(draw, [4, 27, 5], list_point_coords, close=False)
        draw_line_list_points(draw, [10, 8, 11], list_point_coords, close=False)

    im.save(output_filename)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# the script %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':
    setup_image_folder(path_to_images)

    print("Read an image from the webcam...")
    get_image(path_to_images, name_image)

    print("Process the image using Google API...")
    with open("./{}/{}".format(path_to_images, name_image), 'rb') as image:
        print("Detect faces with Google cloud app...")
        faces = detect_face(image)
        # print(faces)
        print("Highlight signigicant points...")
        highlight_faces(image, faces, "./{}/crrt_image_highlighted_{}".format(path_to_images, name_image))

    plt.figure()
    crrt_image = misc.imread("./{}/crrt_image_highlighted_{}".format(path_to_images, name_image))
    plt.imshow(crrt_image)
    plt.show(block=True)
