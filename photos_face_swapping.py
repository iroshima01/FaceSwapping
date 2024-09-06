import cv2
import numpy as np
import dlib
import time
import os
import random
import itertools

ALIGN_POINTS = [17, 18, 19, 20, 21,  # Right eyebrow
                22, 23, 24, 25, 26,  # Left eyebrow
                36, 37, 38, 39, 40, 41,  # Right eye
                42, 43, 44, 45, 46, 47,  # Left eye
                27, 28, 29, 30, 31, 32, 33, 34, 35,  # Nose
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def read_images(n):
    directory_path = "path/to/the/source/images"
    all_files = os.listdir(directory_path)
    image_files = [file for file in all_files if file.endswith(('.png', '.jpg'))]

    detector = dlib.get_frontal_face_detector()
    predictor_path = "path/to /the/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    if n == "multiple":
        for img_paths in image_files:
            img_path = os.path.join(directory_path, img_paths)
            process_image(img_path, n)

    else:
        image_pairs(image_files, directory_path, n)
a

def image_pairs(image_files, directory_path, n):
    for img1_path, img2_path in itertools.permutations(image_files, 2):
        img = cv2.imread(os.path.join(directory_path, img1_path))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(img_gray)
        img2 = cv2.imread(os.path.join(directory_path, img2_path))
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("path/to /the/shape_predictor_68_face_landmarks.dat")
        height, width, channels = img2.shape
        img2_new_face = np.zeros((height, width, channels), np.uint8)
        faces(img, img2, img_gray, img2_gray, detector, predictor, mask, img2_new_face, img1_path, img2_path, n)


def process_image(img_path, n):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("path/to /the/shape_predictor_68_face_landmarks.dat")

    faces = detector(img_gray)
    if len(faces) < 2:
        print(f"Resimde en az iki yüz bulunamadı: {img_path}")
        return

    face1, face2 = faces[:2]

    landmarks_points1, points1, convexhull1 = landmarks_point(img_gray, predictor, face1)
    landmarks_points2, points2, convexhull2 = landmarks_point(img_gray, predictor, face2)

    cv2.fillConvexPoly(mask, convexhull1, 255)
    face_image_1 = cv2.bitwise_and(img, img, mask=mask)

    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull1)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points1)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = process_triangles(triangles, points1)

    height, width, channels = img.shape
    img_new_face = np.zeros((height, width, channels), np.uint8)

    lines_space_mask = np.zeros_like(img_gray)
    lines_space_new_face = np.zeros_like(img)

    for triangle_index in indexes_triangles:
        triangle1 = np.zeros((3, 2), dtype=np.int32)
        triangle2 = np.zeros((3, 2), dtype=np.int32)

        for i in range(3):
            triangle1[i] = landmarks_points1[triangle_index[i]]
            triangle2[i] = landmarks_points2[triangle_index[i]]

        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)

        points = np.array([[p[0] - x, p[1] - y] for p in triangle1], np.int32)
        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2
        cropped_tr2_mask = np.zeros((h, w), np.uint8)
        points2 = np.array([[p[0] - x, p[1] - y] for p in triangle2], np.int32)
        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)


        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)


        img_new_face_rect_area = img_new_face[y: y + h, x: x + w]
        img_new_face_rect_area_gray = cv2.cvtColor(img_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img_new_face_rect_area = cv2.add(img_new_face_rect_area, warped_triangle)
        img_new_face[y: y + h, x: x + w] = img_new_face_rect_area


    img_face_mask = np.zeros_like(img_gray)
    img_head_mask = cv2.fillConvexPoly(img_face_mask, convexhull2, 255)
    img_face_mask = cv2.bitwise_not(img_head_mask)

    img_head_noface = cv2.bitwise_and(img, img, mask=img_face_mask)
    result = cv2.add(img_head_noface, img_new_face)

    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv2.seamlessClone(result, img, img_head_mask, center_face2, cv2.NORMAL_CLONE)

    dir_path_output = "path/to/the/file/for/saving"

    output_path = os.path.join(dir_path_output, f"swapped_{os.path.basename(img_path)}")
    cv2.imwrite(output_path, seamlessclone)
    print(f"Saved: {output_path}")





def faces(img, img2, img_gray, img2_gray, detector, predictor, mask, img2_new_face, img1_path, img2_path, n):
    # face 1
    faces = detector(img_gray)

    for face in faces:
        if n == 1:
            landmarks_points, points, convexhull = landmarks_point(img_gray, predictor, face)
            cv2.fillConvexPoly(mask, convexhull, 255)
            face_image_1 = cv2.bitwise_and(img, img, mask=mask)
        else:
            landmarks1 = predictor(img_gray, face)
            landmarks_points = [(landmarks1.part(n).x, landmarks1.part(n).y) for n in ALIGN_POINTS]
            convexhull = cv2.convexHull(np.array(landmarks_points))
            cv2.fillConvexPoly(mask, convexhull, 255)
            points = np.array(landmarks_points, np.int32)
        # Delaunay triangulation
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        indexes_triangles = process_triangles(triangles, points)

    # Face 2
    faces2 = detector(img2_gray)
    for face in faces2:
        if n == "full_landmarks":
            landmarks_points2, points2, convexhull2 = landmarks_point(img2_gray, predictor, face)
        else:

            landmarks2 = predictor(img2_gray, face)
            landmarks_points2 = [(landmarks2.part(n).x, landmarks2.part(n).y) for n in ALIGN_POINTS]
            convexhull2 = cv2.convexHull(np.array(landmarks_points2))
    lines_space_mask = np.zeros_like(img_gray)
    lines_space_new_face = np.zeros_like(img2)

    # Triangulation of both faces
    for triangle_index in indexes_triangles:
        triangle1 = np.zeros((3, 2), dtype=np.int32)
        for i, index in enumerate(triangle_index):
            triangle1[i] = landmarks_points[index]

        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)
        points = np.array([[pt[0] - x, pt[1] - y] for pt in triangle1], dtype=np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        # Lines space
        for i in range(len(triangle1)):
            start_point = triangle1[i]
            end_point = triangle1[(i + 1) % 3]
            cv2.line(lines_space_mask, tuple(start_point), tuple(end_point), 255)
        lines_space = cv2.bitwise_and(img, img, mask=lines_space_mask)

        # Triangulation of second face
        triangle2 = np.zeros((3, 2), dtype=np.int32)
        for i, index in enumerate(triangle_index):
            triangle2[i] = landmarks_points2[index]

        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([[pt[0] - x, pt[1] - y] for pt in triangle2], dtype=np.int32)
        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstructing destination face
        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

    swap_faces(img2_new_face, img2_gray, convexhull2, img2, img, img1_path, img2_path, n)


def landmarks_point(img_gray, predictor, face):
    landmarks = predictor(img_gray, face)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))

    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    return landmarks_points, points, convexhull


def find_index(points, pt):
    index = np.where((points == pt).all(axis=1))
    return extract_index_nparray(index)


def process_triangles(triangles, points):
    indexes_triangles = []
    for t in triangles:
        indexes = [find_index(points, (t[i], t[i + 1])) for i in range(0, 6, 2)]
        if None not in indexes:
            indexes_triangles.append(indexes)
    return indexes_triangles


def swap_faces(img2_new_face, img2_gray, convexhull2, img2, img, img1_path, img2_path, n):
    img2_new_face_gray = cv2.cvtColor(img2_new_face, cv2.COLOR_BGR2GRAY)
    _, mask_newface = cv2.threshold(img2_new_face_gray, 1, 255, cv2.THRESH_BINARY_INV)
    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)

    img2_head_noface = cv2.bitwise_and(img2, img2, mask=mask_newface)
    result = cv2.add(img2_head_noface, img2_new_face)

    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
    save_images(seamlessclone, img, img2, result, img1_path, img2_path, n)


def save_images(seamlessclone, img, img2, result, img1_path, img2_path, n):

    height = max(img.shape[0], img2.shape[0], result.shape[0])
    img = cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))
    img2 = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height))
    seamlessclone = cv2.resize(seamlessclone, (int(seamlessclone.shape[1] * height / seamlessclone.shape[0]), height))

    if n == "full_landmarks":
        dir_path_output = "path/to/the/file/for/outputs"
    if n == "half_landmarks":
        dir_path_output = "path/to/the/file/for/outputs"

    combined_image = cv2.hconcat([img, seamlessclone, img2])
    combined_result_filename = f"a_{img1_path[:-4]}_and_{img2_path[:-4]}.jpg"
    combined_output_path = os.path.join(dir_path_output, combined_result_filename)
    print(f"Saved: {combined_output_path}")

    cv2.imwrite(combined_output_path, combined_image)


def main(mode):
    read_images(mode)


main("multiple")
