from node_class import Node
from tree_class import Tree

"""
Pseudocode:
1. Take a string and determine the relevant frequencies of the characters
2. Build and sort a list of tuples from lowest to highest frequencies
3. Build the Huffman Tree by assigning a binary code to each letter, using shorter codes for the more frequent letters
4. Trim the Huffman Tree (remove the frequencies from the previously built tree)
5. Encode the text into its compressed form
"""
def index(data,char) :
    indice=[]
    for i, c in enumerate(data):
        if c == char :
            indice.append(i)
    return indice
def return_frequency(data):
    # Take a string and determine the relevant frequencies of the characters
    frequency = {}
    for indice,char in enumerate(data):
           if char not in frequency :
              frequency[char]=(data.count(char),index(data,char))
    # Build and sort a list of tuples from lowest to highest frequencies
    frequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    return frequency


# A helper function to the build_tree()
def sort_values(nodes_list, node):
    char1,(node_value,lst) = node.value
    index = 0
    max_index = len(nodes_list)

    while True:
        if index == max_index:
            nodes_list.append(node)
            return
        char2,(current_val,l) = nodes_list[index].value
        if current_val <= node_value:
            nodes_list.insert(index, node)
            return
        index += 1


# Build a Huffman Tree: nodes are stored in list with their values (frequencies) in descending order.
# Two nodes with the lowest frequencies form a tree node. That node gets pushed back into the list and the process repeats
def build_tree(data):
    lst = return_frequency(data)
    nodes_list = []
    for i,node_value in enumerate(lst):
        node = Node(node_value)
        nodes_list.append(node)
        #print(nodes_list[i].value)
    i=0
    while len(nodes_list) != 1:
        first_node = nodes_list.pop()
        second_node = nodes_list.pop()
        char1, (val1,lst1) = first_node.value
        char2, (val2,lst2) = second_node.value
        node = Node((char1 +" "+ char2,(val1 + val2,lst1)))
        node.set_left_child(second_node)
        node.set_right_child(first_node)
        sort_values(nodes_list, node)


    root = nodes_list[0]
    tree = Tree()
    tree.root = root
    return tree,lst

# the function traverses over the huffman tree and returns a dictionary with letter as keys and binary value and value.
# function get_codes() is for encoding purposes
def get_codes(root):
    if root is None:
        return {}
    characters,(frequency ,lst) = root.value
    char_dict = dict([(i, '') for i in str.split(characters)])

    left_branch = get_codes(root.get_left_child())

    for key, value in left_branch.items():
        char_dict[key] += '0' + left_branch[key]

    right_branch = get_codes(root.get_right_child())

    for key, value in right_branch.items():
        char_dict[key] += '1' + right_branch[key]

    return char_dict


# when we've got the dictionary of binary values and huffman tree, tree encoding is simple
def huffman_encoding_func(data):
    if data == []:
        return None, ''
    tree,freq = build_tree(data)
    dict = get_codes(tree.root)
    codes = ''
    for char in dict:
        codes += dict[char]
    return codes,freq


# The function traverses over the encoded data and checks if a certain piece of binary code could actually be a letter
def huffman_decoding_func(data, freq):
    if data == '':
        return ''
    maximum=0
    for char,(val,lst) in freq :
        if max(lst) > maximum :
            maximum = max(lst)
    s = ['' for e in range(maximum+1)]
    for char,(val,lst) in freq :
        for e in lst :
            s[e]=char

    return s

"""code,freq = huffman_encoding_func(["255","44","255","255","1","1"])
print(code)
sortie=huffman_decoding_func(code,freq)
print(sortie)"""
