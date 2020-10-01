"""
We're gonna take a 10 x 10 grid of squares.
Obstacles are black squares.
Objects defined by shape, size, color.
Each square gets an x, y coordinate.
Return list of occupied grids using computer vision.
Find minimimum path between starting object and matching object using a star search.
"""
import cv2
import numpy
import skimage.measure.compare_ssim
import time

class A_StarSearch:
    """
    A* Search algorithm implementation to find the minimum path between 2 points.
    """
    def astar(self, m, startp, endp):
        w, h = 10, 10
        sx, sy = startp
        ex, ey = endp
        node = [None, sx, sy, 0, abs(ex-sx)+abs(ey-sy)] 
        closeList = [node]
        createdList = {}
        createdList[sy*w+sx] = node
        k=0
        while(closeList):
            node = closeList.pop(0)
            x = node[1]
            y = node[2]
            l = node[3] + 1
            k += 1
            if k != 0:
                neighbours = ((x, y+1), (x, y-1), (x+1, y), (x-1, y))
            else:
                neighbours = ((x+1, y), (x-1, y), (x, y+1), (x, y-1))
            for nx,ny in neighbours:
                if nx==ex and ny==ey:
                    path = [(ex, ey)]
                    while node:
                        path.append((node[1], node[2]))
                        node = node[0]
                    return list(reversed(path))
                if 0 <= nx < w and 0 <= ny < h and m[ny][nx] == 0:
                    if ny * w + nx not in createdList:
                        nn = (node, nx, ny, l, l+abs(nx-ex)+abs(ny-ey))
                        createdList[ny*w+nx] = nn
                        nni = len(closeList)
                        closeList.append(nn)
                        while nni:
                            i = (nni-1) >> 1
                            if closeList[i][4] > nn[4]:
                                closeList[i], closeList[nni] = nn, closeList[i]
                                nni = i
                            else:
                                break
        return list()

class ProcessImage:
    """
    DOCSTRING
    """
    def __call__(self):
        self.main('images/test_image1.jpg')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def main(self, image_filename):
        """
        Returns:
            - List of tuples which is the coordinates for occupied grid.
            - Dictionary with information of path.
        """
        occupied_grids = list()
        planned_path = {}
        image = cv2.imread(image_filename)
        (winW, winH) = (60, 60)
        obstacles = list()
        index = [1, 1]
        blank_image = numpy.zeros((60, 60, 3), numpy.uint8)
        list_images = [[blank_image for i in range(10)] for i in range(10)]
        maze = [[0 for i in range(10)] for i in range(10)]
        for (x, y, window) in Traversal.sliding_window(image, stepSize=60, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            clone = image.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            crop_img = image[x:x + winW, y:y + winH]
            list_images[index[0]-1][index[1]-1] = crop_img.copy()
            average_color_per_row = numpy.average(crop_img, axis=0)
            average_color = numpy.average(average_color_per_row, axis=0)
            average_color = numpy.uint8(average_color)
            if (any(i <= 240 for i in average_color)):
                maze[index[1]-1][index[0]-1] = 1
                occupied_grids.append(tuple(index))
            if (any(i <= 20 for i in average_color)):
                obstacles.append(tuple(index))
            cv2.imshow('Window', clone)
            cv2.waitKey(1)
            time.sleep(0.025)
            index[1] = index[1] + 1
            if(index[1]>10):
                index[0] = index[0] + 1
                index[1] = 1
        list_colored_grids = [n for n in occupied_grids if n not in obstacles]
        for startimage in list_colored_grids:
            key_startimage = startimage
            img1 = list_images[startimage[0]-1][startimage[1]-1]
            for grid in [n for n in list_colored_grids  if n != startimage]:
                img = 	list_images[grid[0]-1][grid[1]-1]
                image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                image2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                s = compare_ssim(image, image2)
                if s > 0.9:
                    result = A_StarSearch.astar(
                        maze, (startimage[0]-1, startimage[1]-1), (grid[0]-1, grid[1]-1))
                    list2 = list()
                    for t in result:
                        x, y = t[0], t[1]
                        list2.append(tuple((x+1, y+1)))
                        result = list(list2[1:-1])
                    if not result:
                        planned_path[startimage] = list(['NO PATH', [], 0])
                    planned_path[startimage] = list([str(grid), result, len(result)+1])
        for obj in list_colored_grids:
            if not(planned_path.has_key(obj)):
                planned_path[obj] = list(["NO MATCH", [], 0])
        return occupied_grids, planned_path

class Traversal:
    """
    Traversing through the image to perform image processing
    """
    def sliding_window(self, image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y: y+windowSize[1], x: x+windowSize[0]])

if __name__ == '__main__':
    occupied_grids, planned_path = ProcessImage.main('images/test_image3.jpg')
    print('Occupied Grids:')
    print(occupied_grids)
    print('Planned Path:')
    print(planned_path)
