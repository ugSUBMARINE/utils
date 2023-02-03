# utils
small scripts for different needs

`grid.py` can be used to extract the values from a heat map, which is only available as an image, and then output them as an available numpy array. This data can then be stored in a tsv file. Required inputs are the file path `hm_path`, the number of rectangles in the heat map - both in x `x_rect` and y `y_rect` direction and the orientation of the colorbar `cb_vert` of the heat map.

When parameters are set, use `python3 grid.py`to run the program. Here one first needs to click (left mouse button) on the left upper corner and on the right lower corner of the heat map to get its position. Use a right click to erase the clicks and to start again. The found contour will be displayed. Close the window again with any key on the keyboard. In the following the spots which are used to get the color of each field will be displayed. Close this again. After that close the window by pressing any key on the keyboard. Next, the same has to be done for the colorbar of the heatmap.
In the end the heat map will be displayed as matplotlib figure.
