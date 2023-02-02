from itertools import product
import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_map_coords(img_path: str):
    """captures the click coordinates - to select for contour search start at left
    upper corner and end at right lower corner
    right click deletes the so far captured coordinates
    :parameter
        - img_path:
          file path to the image of interest
    :return
        - img_sele
          the cropped version of the original image
    """
    coords = []

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coords.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            coords.clear()

    # read and show image
    img = cv2.imread(img_path)
    cv2.imshow("map selection", img)
    # get clicked coordinates
    cv2.setMouseCallback("map selection", onMouse)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    coords = np.array(coords)
    # 'crop' image
    img_sele = img[coords[0][1] : coords[1, 1], coords[0][0] : coords[1][0]]
    return img_sele


def get_map(
    img: np.ndarray[tuple[int], np.dtype[int | float]], show_cont: bool = False
):
    """get the biggest contour of the image
    :parameter
        - img:
          read image of the heat map
        - show_cont:
          True to show the image with the biggest contour in green
    :return
        - img:
          the read image
        - x, y:
          x, y coordinate of the start of the biggest contour
        - w, h:
          width and hight of the biggest contour
    """
    # convert to gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur to make recognition easier
    blurred = cv2.blur(gray_img, (5, 5))
    # get all contours in the image
    thresh = cv2.adaptiveThreshold(blurred, 255, 1, 1, 11, 2)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # find the biggest contour by the area
    c = max(contours, key=cv2.contourArea)
    # get the coordinates of the origin and the size
    x, y, w, h = cv2.boundingRect(c)
    if show_cont:
        # draw the biggest contour in green
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("rectangle", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img, x, y, w, h


def get_spaces(start: int, add: int, num: int):
    """get the x or y coordinates needed to construct center coordinates
    :parameter
        - start:
          x or y coordinate of the start of the heat map
        - add:
          width or height of the heat map
        - num:
          number of rectangles in the heat map
    :return
        - gs
          all x or y coordinates needed
    """
    gs_len = start + add
    # number of needed rectangles
    gs_part = gs_len / num
    # to subtract from first and last so points end up in the middle of the rectangles
    gs_half = gs_part * 0.5
    # array of center coordinates for one direction (x or y)
    gs = np.linspace(start + gs_half, gs_len - gs_half, num)
    return gs


def get_vals(
    img: np.ndarray[tuple[int, int, int], np.dtype[int]],
    x: int,
    y: int,
    w: int,
    h: int,
    col: int,
    row: int,
    expand=3,
):
    """gets the mean color values of selected rectangle around the center
    :parameter
        - img:
          the read image
        - x, y:
          x, y coordinate of the start of the biggest contour
        - w, h:
          width and hight of the biggest contour
        - col:
          number of columns in the heat map
        - row:
          number of rows in the heat map
        - expand:
          x and y direction expansion of the selected rectangle (relative to the center)
    :return
        - values
          color values for each rectangle in the heat map
    """
    # all x and y coordinates
    xs = get_spaces(x, w, col).astype(int)
    ys = get_spaces(y, h, row).astype(int)

    # creates all x-y coordinate combinations
    coord_combs = list(product(np.arange(len(xs)), np.arange(len(ys))))

    values = []
    # get the mean value of the center +/- 3 pixel of the center as value
    for i in coord_combs:
        ix = xs[i[0]]
        iy = ys[i[1]]
        ix_start = ix - expand
        ix_end = ix + expand
        iy_start = iy - expand
        iy_end = iy + expand
        values.append(
            [
                *i,
                *np.mean(
                    np.mean(img[iy_start:iy_end, ix_start:ix_end], axis=1), axis=0
                ),
            ]
        )
        cv2.rectangle(img, (ix_start, iy_start), (ix_end, iy_end), (0, 255, 0), 1)
    values = np.asarray(values)
    cv2.imshow("-", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return values


def dist_calc(
    arr1: np.ndarray[tuple[int, int], np.dtype[int | float]],
    arr2: np.ndarray[tuple[int, int], np.dtype[int | float]],
) -> np.ndarray[tuple[int, int], np.dtype[float]]:
    """
    calculates euclidean distances between all points in two k-dimensional arrays
    'arr1' and 'arr2'
        :parameter
            - arr1: N x k array
            - arr2: M x k array
        :return
            - dist: M x N array with pairwise distances
    """
    norm_1 = np.sum(arr1 * arr1, axis=1).reshape(1, -1)
    norm_2 = np.sum(arr2 * arr2, axis=1).reshape(-1, 1)

    dist = (norm_1 + norm_2) - 2.0 * np.dot(arr2, arr1.T)
    # necessary due to limited numerical accuracy
    dist[dist < 1.0e-11] = 0.0

    return np.sqrt(dist)


def retriev_values(
    hm_path: str,
    x_rect: int,
    y_rect: int,
    cm_img_vals: np.ndarray[tuple[int], np.dtype[int | float]],
    cb_vert: bool = False,
    th: int | float = 10,
    cmap_: str = "br",
    expand_: int = 3,
):
    """get the true values of the heat map and plot it
    :parameter
        - hm_path:
          file path to the whole heat map
        - x_rect, y_rect:
          number if rectangles in the x and y direction
        - cm_img_vals:
          true values of the colorbar according to the real plot
        - cb_vert:
          set True if the colorbar is oriented vertically
        - th:
          how much the color values can be off
        - cmap_:
          the color map for plotting
        - expand_:
          x and y direction expansion of the selected rectangle (relative to the center)
    :return
        - fill
          array with values like the original heat map

    """
    num_classes = len(cm_img_vals)
    mat_img, mat_x, mat_y, mat_w, mat_h = get_map(get_map_coords(hm_path), True)
    mat_vals = get_vals(
        mat_img, mat_x, mat_y, mat_w, mat_h, x_rect, y_rect, expand=expand_
    )

    cm_img, cm_x, cm_y, cm_w, cm_h = get_map(get_map_coords(hm_path), True)

    if cb_vert:
        col_vals = get_vals(cm_img, cm_x, cm_y, cm_w, cm_h, 1, num_classes)
    else:
        col_vals = get_vals(cm_img, cm_x, cm_y, cm_w, cm_h, num_classes, 1)

    # calculate the closest color for each point in the heat map
    diff = dist_calc(mat_vals[:, -3:], col_vals[:, -3:])
    closest = np.argmin(diff, axis=0)
    deltas = diff[closest, np.arange(diff.shape[1])]
    closest = closest.astype(float)
    closest[deltas > th] = np.nan

    # fill an array with the correct heat map values
    fill = np.empty((y_rect, x_rect))
    for ci, i in enumerate(mat_vals):
        try:
            fill_val = cm_img_vals[int(closest[ci])]
        except ValueError:
            fill_val = np.nan
        fill[int(i[1]), int(i[0])] = fill_val

    cmap = plt.get_cmap(cmap_, num_classes)
    fig, ax = plt.subplots(figsize=(18, 32))
    mat = ax.imshow(fill, cmap=cmap)
    fig.colorbar(mat)
    plt.show()
    return fill


if __name__ == "__main__":

    hm_vals = retriev_values(
        "br_hm.png",
        61,
        20,
        cm_img_vals=np.arange(0, 5, .5),
        cmap_="inferno",
    )
    """
    hm_vals = retriev_values(
        "test_hm.png",
        23,
        18,
        th=40,
        cb_vert=True,
        cm_img_vals=np.arange(0, 24, 1),
        cmap_="inferno",
        expand_ = 3
    )
    """

    """
    offset = 2
    aa_top_bot = list("GAVILMFYWDERHKSTNQCP")
    seq = list("IAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASKVRR")
    seq_len = len(seq)
    positions = np.arange(offset, seq_len + offset)

    with open("new_dataset.tsv", "w+") as nds:
        nds.write("variants\tnum_mutations\tscore\n")
        for ci, i in enumerate(hm_vals):
            for cj, j in enumerate(i):
                score = hm_vals[ci, cj]
                if not np.isnan(score):
                    nds.write(
                        f"{seq[cj]}{cj + offset}{aa_top_bot[ci]}\t1\t{score}\n"
                    )
    """
