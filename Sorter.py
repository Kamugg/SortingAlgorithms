import math
import random
import sys

import pygame


class Sorter:

    def __init__(self, size=400):
        self.arr = []  # Array
        self.colors = {'red': (255, 50), 'green': (140, 150), 'blue': (240, 255)}  # Color shades
        self.SCREEN_X = 1200
        self.SCREEN_Y = 500
        self.data = None  # Data to display on screen
        self.size = size  # Size of the array
        self.active = 0
        self.screen = None
        self.generate_array()
        self.reset_data()
        self.max_y = 300  # Max height of the array
        self.def_font = None

    def generate_array(self):
        self.arr = random.sample([i for i in range(self.size)], self.size)

    def setup_screen(self, algo_name):
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_X, self.SCREEN_Y))
        pygame.display.set_caption(algo_name)
        pygame.display.set_icon(pygame.image.load(r'.\sorter_icon.png'))
        pygame.font.init()
        self.def_font = pygame.font.SysFont('Times New Roman', 15)

    def close_screen(self):
        pygame.quit()
        self.screen = None
        self.def_font = None
        random.shuffle(self.arr)
        self.reset_data()

    def reset_data(self):
        self.data = {'Name': '',
                     'Array Reads': 0,
                     'Array Writes': 0,
                     'Comparisons': 0}

    def bubble_sort(self):
        self.data['Name'] = 'Bubble Sort'
        swap = True
        bound = self.size - 1  # bound is the position of the last ordered element
        while swap:
            swap = False
            for i in range(0, bound):
                self.active = i
                self.render()
                self.data['Array Reads'] += 2
                self.data['Comparisons'] += 1
                if self.arr[i] >= self.arr[i + 1]:
                    self.swap(i, i + 1)
                    swap = True
            bound -= 1

    def odd_even_sort(self):
        self.data['Name'] = 'Odd Even Sort'
        swap = 0  # Number of times there were no swaps
        start = (0, 1)  # Tuple of starting points
        while swap < 2:  # If there aren't swaps two times in a row the algorithm ends
            for start_point in start:
                for i in range(start_point, self.size - 1, 2):  # Check whole array
                    self.active = i
                    self.data['Array Reads'] += 2
                    self.data['Comparisons'] += 1
                    if self.arr[i] > self.arr[i + 1]:
                        swap = 0  # If there wa a swap, the count is resetted
                        self.swap(i, i + 1)
                swap += 1

    def heap_sort(self):
        self.data['Name'] = 'Heap Sort'
        heap_size = 1
        # The heap is created in place so the rendering is more interesting
        for i in range(1, self.size):
            self.active = i
            self.render()
            self.__climb(i)  # The i-th element is added in the heap and 'climbs' it
            heap_size += 1
        while heap_size != 0:
            self.swap(0,
                      heap_size - 1)  # The first element of the heap is the bigger one, and is put at the end of the array
            self.active = heap_size
            self.render()
            heap_size -= 1
            self.__descend(0, heap_size)  # The new root element in the heap descends in his correct position

    def __climb(self, startpos):
        if startpos == 0:  # If position is the root, return
            return
        self.active = startpos
        self.render()
        f_pos = self.__father(startpos)  # Get position of his father
        self.data['Array Reads'] += 2
        self.data['Comparisons'] += 1
        if self.arr[f_pos] >= self.arr[startpos]:  # If father is bigger, the order is correctand the algorithm stops
            return
        self.active = f_pos
        self.render()
        self.swap(f_pos, startpos)  # Otherwise a swap is performed and recursion continues
        self.__climb(f_pos)

    def __father(self, s_pos):
        return int((s_pos - 1) / 2)  # Returns father position

    def __descend(self, s_pos, size):
        sons = self.__sons(s_pos, size)  # Get sons data
        sons.sort(key=lambda x: x[0])  # Order by value
        self.data['Comparisons'] += 1
        self.data['Array Reads'] += 1
        self.active = s_pos
        self.render()
        if not sons or self.arr[s_pos] > sons[-1][0]:  # If no sons or current node is bigger, return
            return
        self.swap(s_pos,
                  sons[-1][1])  # Otherwise swap with biggest son (In this way the order of the heap is preserved)
        self.__descend(sons[-1][1], size)

    def __sons(self, s_pos, size):  # Returns an array of (value, position of son), if any
        return [(self.arr[s_pos * 2 + i], s_pos * 2 + i) for i in (1, 2) if s_pos * 2 + i < size]

    def shaker_sort(self):
        self.data['Name'] = 'Shaker Sort'
        bounds = [0, self.size - 1]
        direction = 1  # Tells if we are descending or ascending
        swaps = False
        while abs(bounds[1] - bounds[0]) >= 1:
            if direction == 1:
                swaps = False
            for i in range(bounds[0], bounds[1], direction):
                self.active = i
                self.render()
                self.data[
                    'Comparisons'] += 1  # Just 1 comparison because if first condition is met Python skips the second
                if direction == 1 and self.arr[i] >= self.arr[i + direction]:
                    self.swap(i, i + direction)
                    swaps = True
                if direction == -1 and self.arr[i] <= self.arr[i + direction]:
                    self.swap(i, i + direction)
                    swaps = True
            if not swaps:   # If no swaps were performed, we are done!
                break
            bounds[1] += -1 * direction  # Updating bounds
            bounds.reverse()
            direction *= -1  # Change direction

    def strand_sort(self):
        self.data['Name'] = 'Strand Sort'
        solution = 0  # Length of sorted array
        sub_list = 1  # Length of sublist
        while solution != self.size:
            for i in range(sub_list, len(self.arr)):  # Builds sublist
                self.active = i
                self.data['Array Reads'] += 2
                self.data['Comparisons'] += 1
                self.render()
                if self.arr[i] > self.arr[sub_list - 1]:  # Item is added only if greater than last sublist item
                    self.swap(sub_list, i)
                    sub_list += 1
            self.__merge((0, solution), (solution, sub_list))  # Sublist is then merged with sorted array
            solution = sub_list  # Sorted array's length is updated
            sub_list += 1

    def selection_sort(self):
        self.data['Name'] = 'Selection Sort'
        for start in range(0, self.size - 1):
            minpos = start
            for i in range(start, len(self.arr)):  # Finds the minimum
                self.active = i
                self.data['Array Reads'] += 1
                self.data['Comparisons'] += 1
                if self.arr[i] <= self.arr[minpos]:
                    minpos = i
                self.render()
            self.swap(minpos, start)  # Put the minimum in correct position

    def insertion_sort(self, start=0, end=-1):
        if end == -1:
            self.data['Name'] = 'Insertion Sort'
            end = len(self.arr)
        for j in range(start + 1, end):
            index = j
            # For each element swap until it reaches the correct position
            while index > start and self.arr[index - 1] > self.arr[index]:
                self.data['Comparisons'] += 1
                self.swap(index, index - 1)
                index -= 1
            self.render()

    def trippel_sort(self, start=0, end=-1):  # This one is kind of black magic
        if end == -1:
            self.data['Name'] = 'Trippel Sort (Stooge Sort)'
            end = self.size
        if end - start <= 1:
            return
        if end - start == 2:
            self.data['Comparisons'] += 1
            if self.arr[start] > self.arr[start + 1]:
                self.swap(start, start + 1)
            return
        break_point = int((end - start) / 3)
        self.trippel_sort(start, end - break_point)
        self.trippel_sort(start + break_point, end)
        self.trippel_sort(start, end - break_point)

    def bitonic_sort(self, start=0, end=-1):
        if end == -1:  # If first iteration checks that array's length is 2^k
            end = len(self.arr)
            n = math.log(end - start, 2)
            if n != int(n):
                print('Error: Array length must be a power of two!')
                return
            self.data['Name'] = 'Bitonic Sort'
        if end - start == 2:
            return
        mid = int((end + start) / 2)
        self.bitonic_sort(start, mid)
        self.bitonic_sort(mid, end)
        self.__build_bitonic((start, mid), 'asc')
        self.__build_bitonic((mid, end), 'desc')
        if start == 0 and end == self.size:
            self.__build_bitonic((start, end), 'asc')

    def __build_bitonic(self, bounds, direction):
        distance = int((bounds[1] - bounds[0]) / 2)  # Distance between comparisons
        operation = []  # Containsthe valued already compared
        while distance > 0:
            for i in range(bounds[0], bounds[1] - distance):
                self.active = i
                self.render()
                if i not in operation:  # Processed only if the value wasn't already compared
                    self.data['Array Reads'] += 2
                    self.data['Comparisons'] += 1
                    # op contains the comparisons to evaluate based on the direction
                    op = {'asc': self.arr[i] > self.arr[i + distance],
                          'desc': self.arr[i] <= self.arr[i + distance]}
                    if op[direction]:  # If condition is met, values get swapped and appended to 'already processed'
                        self.swap(i, i + distance)
                        operation.append(i + distance)
            distance = int(distance / 2)  # Distance gets halved at each iteration
            operation.clear()

    def merge_sort(self, start=0, end=-1):
        # Only happens if this is the first recursive call
        if end == -1:
            end = len(self.arr)
        self.data['Name'] = 'Merge Sort'
        # Exit condition
        if end - start <= 1:
            return
        mid = int((start + end) / 2)
        self.render()
        self.merge_sort(start, mid)
        self.merge_sort(mid, end)
        self.__merge((start, mid), (mid, end))

    def __merge(self, arr_bound, arr1_bound):
        i, j = arr_bound[0], arr1_bound[0]
        while i != j and i < arr1_bound[1] and j < arr1_bound[1]:
            self.data['Array Reads'] += 2
            self.data['Comparisons'] += 1
            if self.arr[i] <= self.arr[j]:
                self.active = i
                i += 1
            else:
                self.active = j
                self.arr.insert(i, self.arr.pop(j))
                self.data['Array Writes'] += 1
                self.data['Array Reads'] += 1
                j += 1
                i += 1
            self.render()

    def quicksort(self, start=0, end=-1):
        if end == -1:  # Check if first iteration
            end = self.size
            self.data['Name'] = 'Quicksort'
        if end - start <= 1:  # Base case
            return
        mid = self.__find_best_piv(start, end)
        self.data['Array Reads'] += 1
        pivot = self.arr[mid]
        self.swap(start, mid)  # Pivot is moved at the beginning of the sub-array
        shift = 1  # Counts ho many elements in the sub-array are less than the pivot
        for i in range(start + 1, end):  # Modifying array to obtain the {x<pivot, pivot, x>pivot} structure
            self.active = i
            self.data['Array Reads'] += 1
            self.data['Comparisons'] += 1
            self.render()
            if self.arr[i] <= pivot:  # Every element lesser than the pivot is put to its left
                self.data['Array Writes'] += 1  # I suppose that insert() is one read and one write...?
                self.data['Array Reads'] += 1
                self.arr.insert(start, self.arr.pop(i))
                shift += 1
        self.quicksort(start, start + shift)  # Recursive calls on the two new sub-arrays
        self.quicksort(start + shift, end)

    def __find_best_piv(self, start, end):  # Finds best pivot (the one that should generate a 50-50 division)
        mean = 0
        for index, elem in enumerate(self.arr[start:end]):
            self.data['Array Reads'] += 1
            mean += elem  # I could have used python's mean(), but I don't know how it works internally, so upgrading
            # Array Reads was tricky
        mean /= end - start
        min_distance = math.inf
        mean_val_index = start
        for index, elem in enumerate(self.arr[start:end]):  # Finds the value that's nearest to the mean
            self.data['Comparisons'] += 1
            self.data['Array Reads'] += 1
            if abs(elem - mean) < min_distance:
                mean_val_index = start + index
                min_distance = abs(elem - mean)
        return mean_val_index   # Returns its index

    def pancake_sort(self):
        self.data['Name'] = 'Pancake Sort'
        for limit in range(self.size - 1, 0, -1):
            maxpos = 0
            for i in range(limit + 1):  # Find maximum
                self.active = i
                self.data['Array Reads'] += 2
                self.data['Comparisons'] += 1
                if self.arr[i] > self.arr[maxpos]:
                    maxpos = i
                self.render()
            self.__flip(maxpos)  # Max value is now in position 0 of the array
            self.__flip(limit)  # Now is in the correct pos

    def __flip(self, bound):  # Flips array, each element is swapped with its simmetric (from the center)
        for i in range(0, int(bound / 2) + 1):
            self.swap(bound - i, i)

    def shell_sort(self):
        self.data['Name'] = 'Shell Sort'
        gap = int(self.size / 2)
        while gap > 0:
            for i in range(gap, len(self.arr)):  # Keeps swapping until item is inorder or the 0 index is reached
                self.active = i
                self.render()
                start = i
                while start - gap >= 0 and self.arr[start] <= self.arr[start - gap]:
                    self.active = start
                    self.data['Array Reads'] += 2
                    self.data['Comparisons'] += 1
                    self.swap(start, start - gap)
                    start -= gap
            gap = int(gap / 2)

    def comb_sort(self):
        self.data['Name'] = 'Comb Sort'
        gap = int(self.size / 1.25)  # 1.25 is a constant found online that seems to be optimal
        sorted = False
        while not sorted:  # Keeps going until gap == 1 and no elements are swapped
            swaps = False
            for i in range(self.size):
                if i + gap < self.size and self.arr[i] > self.arr[i + gap]:
                    self.data['Array Reads'] += 2
                    self.data['Comparisons'] += 1
                    self.swap(i, i + gap)
                    swaps = True
            if not swaps and gap == 1:
                sorted = True
            if gap != 1:
                gap = int(gap / 1.25)

    def gnome_sort(self):
        self.data['Name'] = 'Gnome Sort'
        pos = 0
        while pos != self.size:
            self.active = pos
            self.data['Comparisons'] += 1
            self.render()
            if pos == 0 or self.arr[pos] >= self.arr[
                pos - 1]:  # Increment position if at the start or if pair is ordered
                pos += 1
            else:  # Otherwie performs a swap
                self.swap(pos - 1, pos)
                pos -= 1

    def tim_sort(self, run=64):
        self.data['Name'] = 'Tim Sort'
        run_list = [(i, i + run) for i in range(0, len(self.arr), run)]  # Contains pairs of length run
        run_list[-1] = (run_list[-1][0], len(self.arr))  # Add last element
        self.__recurse(run_list, 0, len(run_list))

    def __recurse(self, run_list, start, stop):
        if start >= stop:
            return
        if stop - start == 1:  # If only 1 slice, perform insertion sort from pair[0] to pair[1]
            self.insertion_sort(start=run_list[start][0], end=run_list[start][1])
            return
        # Otherwise divides the array in two parts
        mid = int((start + stop) / 2)
        left = (run_list[start][0], run_list[mid - 1][1])
        right = (run_list[mid][0], run_list[stop - 1][1])
        self.__recurse(run_list, start, mid)
        self.__recurse(run_list, mid, stop)
        self.__merge(left, right)  # Sorted slices are then merged together

    def check(self, rend=True):
        srted = True
        for i in range(len(self.arr) - 1):
            self.active = i
            if rend:
                self.render()
            if self.arr[i] > self.arr[i + 1]:
                srted = False
                break
        return srted

    def swap(self, a, b):
        if a == b:
            return
        self.data['Array Reads'] += 2
        self.data['Array Writes'] += 2
        tmp = self.arr[a]
        self.arr[a] = self.arr[b]
        self.arr[b] = tmp
        self.active = b
        self.render()

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        self.screen.fill((0, 0, 0))
        for index, val in enumerate(self.arr):
            if index == self.active:
                color = 'green'
            else:
                color = []
                for col in ['red', 'green', 'blue']:
                    color.append(
                        self.colors[col][0] + int(((self.colors[col][1] - self.colors[col][0]) / self.size) * val))
                color = tuple(color)
            pygame.draw.rect(self.screen, color, (
                index * (self.SCREEN_X / self.size), self.SCREEN_Y - (self.max_y / self.size) * val,
                self.SCREEN_X / self.size,
                (self.max_y / self.size) * val))
        for i, k in enumerate(self.data.keys()):
            ts = self.def_font.render(f'{k}: {self.data[k]}', False, 'white')
            self.screen.blit(ts, (0, i * 20))
        for index, c in enumerate(self.colors.keys()):
            ts = self.def_font.render(f'{c}: From {self.colors[c][0]} to {self.colors[c][1]}', False, 'white')
            self.screen.blit(ts, (0, 80 + index * 20))
        pygame.display.update()
