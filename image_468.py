import cv2
import numpy as np
import dlib
import time
import mediapipe as mp
from alignment import face_alignment

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def swapFace(path_src, path_dst):
    img = cv2.imread(path_src)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    img2 = cv2.imread(path_dst)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    # mp_drawing = mp.solutions.drawing_utils
    # mp_drawing_styles = mp.solutions.drawing_styles

    # my_drawing_specs = mp_drawing.DrawingSpec(color = (0, 255, 0), thickness = 1)

    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Facial landmarks
    result = face_mesh.process(rgb_image)
    height, width, _ = img.shape
    # for facial_landmarks in result.multi_face_landmarks:
    #     for i in range(0, 468):
    #         pt1 = facial_landmarks.landmark[i]
    #         x = int(pt1.x * width)
    #         y = int(pt1.y * height)
    #         cv2.circle(image, (x, y), 5, (100, 100, 0), -1)
    #         print(x,y)
    # cv2.imshow("Image", image)
    # cv2.waitKey(1)


    # Face 1
    for facial_landmarks in result.multi_face_landmarks:
    #   landmarks = predictor(img_gray, face)
        landmarks_points = []
        landmarks_points2 = []
        for n in range(0, 468):
            pt1 = facial_landmarks.landmark[n]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
            landmarks_points.append((x, y))
    #         print(x,y)

            # cv2.circle(img, (x, y), 3, (0, 0, 255), -1)


        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
        cv2.fillConvexPoly(mask, convexhull, 255)

        face_image_1 = cv2.bitwise_and(img, img, mask=mask)

        # Delaunay triangulation
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        # print(len(triangles))
        triangles = np.array(triangles, dtype=np.int32)

        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])


            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)

            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)

            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)

            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)


    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_new_face = np.zeros_like(img2)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    rgb_image = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # Facial landmarks
    result1 = face_mesh.process(rgb_image)
    height1, width1, _ = img2.shape
    # Face 2
    for facial_landmarks in result1.multi_face_landmarks:
#       landmarks = predictor(img2_gray, face)
        landmarks_points2 = []
        for n in range(0, 468):
            pt1 = facial_landmarks.landmark[n]
            x = int(pt1.x * width1)
            y = int(pt1.y * height1)
            landmarks_points2.append((x, y))

        # cv2.circle(img2, (x, y), 3, (0, 255, 0), -1)
        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)

    lines_space_mask = np.zeros_like(img_gray)
    lines_space_new_face = np.zeros_like(img2)
    # Triangulation of both faces
    for triangle_index in indexes_triangles:
        # Triangulation of the first face    
        
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)


        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)


        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                        [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        # Lines space
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
        cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
        lines_space = cv2.bitwise_and(img, img, mask=lines_space_mask)

        # Triangulation of second face
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)


        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

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



    # Face swapped (putting 1st face into 2nd face)
    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)


    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result = cv2.add(img2_head_noface, img2_new_face)

    x, y, w, h = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)

    # cv2.imshow("img", img)
    # cv2.imshow("img2", img2)
    cv2.imshow("seamlessclone", seamlessclone)
    cv2.imwrite("result.jpg", seamlessclone)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    path_src = "D:/Dataset/00000/00606.png"
    cv2.imwrite('src_image.png', cv2.imread(path_src))
    path_alignment = 'alignment_image.png'
    path_dst = "D:/Dataset/00000/00300.png"
    cv2.imwrite('dst_image.png', cv2.imread(path_dst))
    try:
        face_alignment(path_src, path_alignment)
        swapFace(path_alignment, path_dst)
    # except AssertionError:
    #     print("Cannot swap!")
    finally:
        print("Done!")