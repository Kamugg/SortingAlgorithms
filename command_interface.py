import Sorter

command = ['']
sorter = Sorter.Sorter()
functions = {
    'bubblesort': sorter.bubble_sort,
    'mergesort': sorter.merge_sort,
    'oddevensort': sorter.odd_even_sort,
    'heapsort': sorter.heap_sort,
    'shakersort': sorter.shaker_sort,
    'strandsort': sorter.strand_sort,
    'selectionsort': sorter.selection_sort,
    'insertionsort': sorter.insertion_sort,
    'trippelsort': sorter.trippel_sort,
    'bitonicsort': sorter.bitonic_sort,
    'quicksort': sorter.quicksort,
    'pancakesort': sorter.pancake_sort,
    'shellsort': sorter.shell_sort,
    'combsort': sorter.comb_sort,
    'gnomesort': sorter.gnome_sort,
    'timsort': sorter.tim_sort
}
help_message = '''
Welcome to visual sorter! This program will show you most of the sorting algorithms.
This is the command prompt that will let you run the algorithms. The possible commands are:
                  
1) run algorithm_name
                  
Replace algorithm_name with one of these:
                  
%s
                  
2) set size newsize
                  
Modifies the number of elements in the array that's going to be sorted. newsize must be an int between 1 and 1024.
The default size is set to 400. Keep in mind that some algorithms are way slower than others!
                  
3) set color your_color new_bound1 new_bound2
                  
Modifies the shades of your_color in the array; your_color must be red, green or blue.
new_bound1 and new_bound2 must be int between 0 and 255.
For example set color red 100 200 will set the red value of the first element of the array
to 100 and the last one to 200.

4) quit

Closes this prompt

5) help

Shows this message.
'''
while command[0] != 'quit':
    raw_command = input('Insert command (type help to get a list of all possible commands!): ')
    command = raw_command.split()
    if len(command) == 2 and command[0] == 'run':
        if command[1] in functions:
            sorter.setup_screen('Running ' + command[1].capitalize())
            functions[command[1]]()
            sorter.check()
            sorter.close_screen()
        else:
            print('Algorithm not found')
    elif len(command) >= 3 and command[0] == 'set':
        if command[1] == 'size':
            try:
                newsize = int(command[2])
                if not(0 < newsize < 1025):
                    print('Invalid input. New size must be greater than 0 and less than 1025.')
                else:
                    sorter.size = newsize
                    sorter.generate_array()
            except:
                print('Invalid input. New size must be an int.')
        elif command[1] == 'color' and len(command) == 5:
            try:
                if not(0 <= int(command[3]) <= 255) or not(0 <= int(command[4]) <= 255):
                    print('Invalid input: color boundaries must be an int from 0 to 255.')
                else:
                    if command[2] in sorter.colors:
                        sorter.colors[command[2]] = (int(command[3]), int(command[4]))
                    else:
                        print(f'Invalid input. There is no color named \'{command[2]}\'. Use red|blue|green.')
            except ValueError:
                print(f'Invalid input: color boundaries must be an int from 0 to 255.')
        else:
            print(f'Unknown syntax for command \'{command[0]}\'.')
    elif len(command) == 1 and command[0] == 'help':
        print(help_message % '\n'.join(functions.keys()))
    elif command:
        print(f'Command {command[0]} not found.')
