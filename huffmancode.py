# Dan Tran 
# COMP 211 (Professor Krizanc) Final Project
# Huffman Encoding Algorithm
# 12/17/2016

import heapq
import cPickle
import os
import random
import zlib
from math import log

class FullBinaryTree(object):

    '''Implements a full binary tree; each node should have exactly two children,
       left and right, and one parent. For interal nodes left and right are
       are other internal nodes. For leaves, the are both None. All nodes
       have a parent that is an internal node except the root whose parent
       is None. Tree must contain at least one node.'''

    def __init__(self,left=None,right=None,parent=None):

        '''Constructor creates a single node tree as default. Sets
           parent relation if left and right are given.'''

        self.left = left
        self.right = right
        self.parent = parent
        if self.left:
            self.left.set_parent(self)
        if self.right:
            self.right.set_parent(self)

    def set_parent(self,tree):

        self.parent = tree

    def get_parent(self):

        return self.parent

    def is_leaf(self):

        '''Returns true iff node is a leaf'''

        return not self.left and not self.right

    def is_root(self):

        '''Returns true iff node is the root'''

        return not self.parent

    def size(self):

        '''Returns the size of the tree'''

        if self.is_leaf():
            return 1
        else:
            return 1 + self.left.size() + self.right.size()

    def height(self):

        '''Returns the height of the tree'''

        if self.is_leaf():
            return 0
        else:
            return 1 + max((self.left.height(),self.right.height()))

    def lca(self,tree):

        '''Returns the least common answer of self and tree'''

        my_anc = self.list_of_ancestors()
        tree_anc = tree.list_of_ancestors()
        i=0
        while  i<len(my_anc) and i<len(tree_anc) and my_anc[i] == tree_anc[i]:
            i = i+1
        if my_anc[i-1] == tree_anc[i-1]:
            return my_anc[i-1]
        else:
            return None


    def contains(self,tree):

        '''Returns true iff self contains tree as a subtree'''

        if self == tree:
            return True
        elif self.is_leaf():
            return False
        else:
            return self.left.contains(tree) or self.right.contains(tree)

    def list_of_ancestors(self):
        '''Returns list of ancestors including self'''

        if self.is_root():
            return [self]
        else:
            return self.parent.list_of_ancestors() + [self]

    def list_of_leaves(self):

        '''Returns a list of all of the leaves of tree'''

        if self.is_leaf():
            return [self]
        else:
            return self.left.list_of_leaves()+self.right.list_of_leaves()


class HuffmanTree(FullBinaryTree):
    
    def __init__(self,left=None,right=None,parent=None,symbol=None,prob=None,code=''):
        '''Constructor for a single node HuffmanTree as a default.
           Sets a parent relation if left and right are given.'''
        FullBinaryTree.__init__(self,left,right,parent)
        self.symbol = symbol
        self.prob = prob
        self.code = code


    def set_code(self,newCode):
        '''Sets new code for the tree'''

        self.code = newCode

    
    def __cmp__(self,other):
        '''Compares trees according to their probabilities (relative frequencies)'''
        
        return cmp(self.prob, other.prob)

    def get_codeword(self):
        
        '''Returns the binary string (made up of 0s and 1s)
           created by concatenating the code values on the path from
           the root to self (not including the root code value).'''

        lst = self.list_of_ancestors()
        codeword = ''
        for i in range(len(lst)):
            codeword += lst[i].code
        return codeword

    def get_symbol(self,symbol):
        
        '''Returns the leaf node in the tree containing 
           the given symbol if such a leaf exists.'''

        for tree in self.list_of_leaves():
            if symbol == tree.symbol:
                return tree
        return None
    
def frequency_count(text):
    
    '''Given the name of the text file, reads the file and returns
       two lists representing symbols and their relative frequencies'''
    
    dic = {}
    symbol_lst = []
    freq_lst = []
    f = open(text,'r')
    for line in f:
        for symbol in line:
            if symbol not in dic:
                dic[symbol] = 1
            else:
                dic[symbol] += 1
    f.close()
    for symbol,freq in dic.items():
        symbol_lst.append(symbol)
        freq_lst.append(freq)
    return symbol_lst,freq_lst

def create_huffman_tree(symbol_lst,freq_lst):
    
        '''Given list of symbols and their relative frequencies list,
           creates a huffman tree object. If there's only one element in the lists, manually create
           a node with arbitrary code (in this case, 0)'''
        if (len(symbol_lst) > 1):
            nodes = [HuffmanTree(symbol=symbol_lst[i],prob=freq_lst[i],code = '') for i in range(len(symbol_lst))]
            heapq.heapify(nodes)
            while len(nodes) > 1:
                tree1 = heapq.heappop(nodes)
                tree2 = heapq.heappop(nodes)
                tree1.set_code('0')
                tree2.set_code('1')
                parent = HuffmanTree(left=tree1,right=tree2,prob=tree1.prob+tree2.prob)
                heapq.heappush(nodes,parent)
            return nodes[0]

        else:
            node = HuffmanTree(symbol = symbol_lst[0], code = '0', prob = freq_lst[0])
            return node

def huffman_dictionary(huffmanTree):

        '''Given a huffman tree object,
           creates a dictionary in a form of {symbol:codeword}'''
        huffmancode = {}
        for tree in huffmanTree.list_of_leaves():
            huffmancode[tree.symbol] = tree.get_codeword()
        return huffmancode

def huffman_decoder(huffmanCode, binStr):
    
        '''Given huffman code dictionary and a binary string, rebuilts
           the binary string into original string of text'''
        
        reverseDict = {}
        rebuiltStr = ''
        for i in huffmanCode:
            reverseDict[huffmanCode[i]] = i
        begIndex = 0
        curIndex = 0
        while curIndex < len(binStr) + 1:
            try:
                rebuiltStr += reverseDict[str(binStr[begIndex:curIndex])] 
                begIndex = curIndex 
            except KeyError:
                curIndex += 1 
        return rebuiltStr


def binary2char(string):

    '''Returns character encoded version of a binary string.
       Note: padded to be divisible by 8 with pad length as first char.'''

    pad = 8 - len(string)%8
    string = string+pad*'0'
    out = str(pad)+''.join([chr(int(string[i:i+8],2))
                            for i in range(0,len(string),8)])
    return out

def char2binary(string):

    '''Returns binary string represented by a character string.
       Assumes first char represents number of pad bits.'''

    pad = int(string[0])
    out = ''.join([(10-len(bin(ord(char))))*'0' + bin(ord(char))[2:] for
                    char in string[1:]])
    return out[:-1*pad]

def encode(infile, outfile):
    
    '''Given the name of an input file and output file, compresses the input file using huffman
    dictionary and binary2char function. Stores representation of huffman dictionary using
    cPickle module. If the size of the input file is empty, the output file is going to be empty too.'''
    
    wholeDoc = open(infile, 'r')
    outputfile = open(outfile, 'w')
    output = ''
    print('Encoding %s into %s...'%(infile, outfile))
    if (os.path.getsize(infile) == 0):
        outputfile.close()
    else:
        huffmancode = huffman_dictionary(create_huffman_tree(frequency_count(infile)[0],frequency_count(infile)[1]))
        for line in wholeDoc:
            for ch in line:
                output += huffmancode[ch]
        outputfile.write(binary2char(output))
        outputfile.close()
        with open('dictionary.pickle', 'wb') as f:
            cPickle.dump(huffmancode, f)
    print('Original Size %s bytes. Compressed Size: %s bytes'%(os.path.getsize(infile), os.path.getsize(outfile)))


def decode(infile, outfile):
    
    '''Given the name of the compressed file decodes it using huffman dictionary stored
       in a pickle file and char2binary function'''
    f = open(infile, 'r')
    fString = f.read()
    outputfile = open(outfile, 'w')
    print('Decoding %s into %s...'%(infile, outfile))
    if (os.path.getsize(infile) == 0):
        outputfile.close()
    else:
        with open('dictionary.pickle', 'rb') as f:
            huffmancode = cPickle.load(f)
        char2bin = char2binary(fString)
        outputfile.write(huffman_decoder(huffmancode, str(char2bin)))
        outputfile.close()
    print('Compressed Size: %s bytes. Decompressed Size: %s bytes'%(os.path.getsize(infile), os.path.getsize(outfile)))


def create_file(givenalphabet, n):
    
    '''Creates new file named 'textcase.txt' consisting of n
       random symbols chosen randomly from a given alphabet'''
    
    newfile = open('testcase.txt', 'w')
    alphabet = list(givenalphabet)
    text = ''
    for i in range(n):
        text += random.choice(alphabet)
    newfile.write(text)
    newfile.close


def comparecontent(file1, file2):
    
    '''Checks two .txt files to see if they have the same content'''
    
    filecontent1 = open(file1, 'r')
    filecontent2 = open(file2, 'r')
    f1Str = filecontent1.read()
    f2Str = filecontent2.read()
    if f1Str == f2Str:
            print 'Two files have identical content'
    else:
            print 'They\'re different'

def entropy(inputfile):
    
    '''Computes entropy of a given file from the relative frequencies of symbols it contains'''
    
    freqs = frequency_count(inputfile)[1]
    problist = []
    theSum = 0
    entropy = 0
    for i in freqs:
        theSum = theSum + i
    for count in freqs:
        problist.append(float(count)/theSum)
    for c in problist:
        entropy += c * log(1/c,2)
    print('Entropy of file %s is %f'%(inputfile, entropy))
    return entropy
    

def zlibcompress(infile, outfile):
    
    '''Compresses the file using zlib python module and creates a file called 'zlib.txt' containing the compressed data'''
    
    f = open(infile, 'r')
    fstr = f.read()
    zlibcompressed = zlib.compress(fstr)
    outputfile = open(outfile, 'w')
    outputfile.write(zlibcompressed)
    outputfile.close()
    size = os.path.getsize(outfile)
    print('Compressing using zlib... The size of the compressed file is: %s bytes'%size)


def main():
    
    '''Main method. Change file names to use'''
    
    encode('sample.txt','sample_encoded.txt')
    decode('sample_encoded.txt','sample_decoded.txt')
    comparecontent('sample.txt','sample_decoded.txt')
    zlibcompress('sample.txt', 'sample_zlib.txt')
    entropy('sample.txt')
    print '-------'
    encode('fasta.txt','fasta_encoded.txt')
    decode('fasta_encoded.txt','fasta_decoded.txt')
    comparecontent('fasta.txt','fasta_decoded.txt')
    zlibcompress('fasta.txt', 'fasta_zlib.txt')
    entropy('fasta.txt')
    print '-------'
    encode('allAs.txt','allAs_encoded.txt')
    decode('allAs_encoded.txt','allAs_decoded.txt')
    comparecontent('allAs.txt','allAs_decoded.txt')
    zlibcompress('allAs.txt', 'allAs_zlib.txt')    
    entropy('allAs.txt')
    print '-------'    
    encode('random.txt','random_encoded.txt')
    decode('random_encoded.txt','random_decoded.txt')
    comparecontent('random.txt','random_decoded.txt')
    zlibcompress('random.txt', 'random_zlib.txt')
    entropy('random.txt')
    print '-------'
    encode('empty.txt','empty_encoded.txt')
    decode('empty_encoded.txt','empty_decoded.txt')
    comparecontent('empty.txt','empty_decoded.txt')
    zlibcompress('empty.txt', 'empty_zlib.txt')
    entropy('empty.txt')
    

#-------------------------------------------------------RESULTS-------------------------------------------------------#
# FILE NAME:       SIZE:       COMPRESSED SIZE:    PERCENTAGE(HUFFMAN):    ZIP SIZE:    PERCENTAGE(ZLIB)     ENTROPY
# sample.txt    46975 bytes      27913 bytes         40.58%                  18968           59.62%       4.70968590567
# fasta.txt      7060 bytes       1961 bytes         72.22%                  2073            70.64%       2.11422675223                 
# allAs.txt     20000 bytes       2502 bytes         87.49%                  43              99.79%       0.0
# random.txt    20000 bytes      12019 bytes         39.91%                  12628           36.86%       4.7541182661
# empty.txt         0 bytes          0 bytes           -                     8                 -          0.0
#
# Based on the data, one can notice a negative correlation between entropy and the effectiveness of compression algorithms. Huffman algorithm did the best
# with fasta and allA's and relatively bad with random.txt because fasta contains at most 10 symbols and so the Huffman Tree object is smaller and allA's
# only contains one node.Huffman algorithm outperformed zlib's compression on fasta.txt, random.txt and empty.txt. When it came to a real text
# and allA's, the zlib was better. I think the huffman algorithm didn't do as good on the sample.txt and random.txt because it variety of symbols was
# relatively great and so entropy was high too. The Huffman Tree had many nodes and since the text was chaotic the compression wasn't very effective.


