"""
bow.py — Bag-of-Words (BoW) Feature Extractor for Java Patch Files

This script reads Java patch files (e.g., from JDT), extracts deleted source
code lines, tokenizes them into individual words (filtering out Java keywords
and comments), and builds a Bag-of-Words vocabulary. It then generates a CSV
file where each row corresponds to a patch file and each column represents a
word from the BoW vocabulary, recording word frequency counts per patch.

Usage:
    python bow.py

Output:
    bow.csv — A CSV file with BoW feature vectors for each patch file.

Note:
    The patch directory path is hardcoded (e.g., './jdt/patch/*.patch').
    Modify the `path` variable to point to the desired project's patches.
"""

import sys
import glob
import errno
import re
from sklearn.feature_extraction.text import CountVectorizer
import csv


def word_extraction(sentence):
    """
    Tokenize a line of Java source code into individual words.

    Splits the sentence on whitespace, punctuation, and special characters,
    then removes common Java keywords (e.g., 'abstract', 'class', 'public')
    to retain only meaningful identifiers and tokens.

    Args:
        sentence (str): A single line of Java source code.

    Returns:
        list[str]: A list of cleaned, non-keyword tokens extracted from
                   the sentence.
    """
    # Java keywords and common tokens to exclude from the vocabulary
    keywords = ['abstract', 'assert', 'boolean', 'byte', 'catch', 'char', 'class', 'const', 'double', 'enum', 'exports', 'extends', 'final', 'finally', 'float', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', 'package', 'private', 'protected', 'public', 'short', 'static', 'super', 'this', 'throw', 'throws', 'try', 'void', 'util', 'java']    

    # Split on whitespace, operators, delimiters, and numeric literals
    words = re.split(r"\s+|\.|\(|\)|\{|\}|\[|\]|\;|\=|\!|\&|\||\+|\-|\*|\%|\>|\<|\?|\:|\"|\#|\'|\,|\^|\\n|\\t|\d+", sentence) 

    # Filter out keywords and empty strings
    cleaned_text = [w for w in words if w not in keywords and w != '']    
    return cleaned_text


def tokenize(sentences):
    """
    Tokenize a list of sentences into a flat list of words.

    Iterates over each sentence, applies word_extraction(), and
    collects all extracted tokens into a single list.

    Args:
        sentences (list[str]): A list of source code lines to tokenize.

    Returns:
        list[str]: A flat list of all extracted tokens from all sentences.
    """
    words = []    
    for sentence in sentences:        
        w = word_extraction(sentence)        
        words.extend(w)            
        
    return words


def generate_bow(allsentences):
    """
    Generate a Bag-of-Words dictionary from a list of source code lines.

    Tokenizes all sentences, fits a CountVectorizer to build a vocabulary,
    and filters out words with a vocabulary index below 4 (i.e., very common
    or short words that are not informative features).

    Args:
        allsentences (list[str]): A list of source code lines.

    Returns:
        dict or None: A dictionary mapping words to their vocabulary indices,
                      or None if the vocabulary is empty.
    """
    vocab = tokenize(allsentences)
    if vocab:
        # Build vocabulary using scikit-learn's CountVectorizer
        vectorizer = CountVectorizer()
        vectorizer.fit_transform(vocab).todense()
        keyword_dict = vectorizer.vocabulary_

        # Remove low-index (very common) words from the vocabulary
        for key in [key for key in keyword_dict if keyword_dict[key] < 4]: del keyword_dict[key] 
        
        return keyword_dict


def remove_comments(line, sep):
    """
    Strip inline comments from a line of source code.

    Finds the first occurrence of any comment separator (e.g., '//')
    and removes everything from that point onward.

    Args:
        line (str): A line of source code.
        sep (str): The comment separator string to look for.

    Returns:
        str: The line with inline comments removed, stripped of
             leading/trailing whitespace.
    """
    for s in sep:
        i = line.find(s)
        if i >= 0:
            line = line[:i]
    return line.strip()


# =============================================================================
# Phase 1: Build the global BoW vocabulary from all patch files
# =============================================================================
allsentences = []
bow_list = []

# Path to the patch files — change this to target a different project
path = './jdt/patch/*.patch'   
files = glob.glob(path)   

for name in files:
    try:
        with open(name, errors='ignore') as f:
            Lines = f.readlines()
            for line in Lines:
                # Only process deleted lines (lines starting with '-')
                if line.startswith('-'):
                    line = line[1:].strip()
                    # Skip diff headers, block comments, and single-line comments
                    if not (line.startswith('--') or line.startswith('*') or line.startswith('/*') or line.startswith('*/') or line.startswith('//')):
                        allsentences.append(remove_comments(line, '//'))
            
    except IOError as exc:
        # Silently skip directories encountered during glob iteration
        if exc.errno != errno.EISDIR:
            raise

# Generate the global BoW vocabulary from all collected sentences
bow = generate_bow(allsentences)

if bow is not None:
    bow_list.append(bow)

# =============================================================================
# Phase 2: Build per-patch BoW feature vectors using the global vocabulary
# =============================================================================

# Create a template row with all vocabulary keys initialized to 0
keys = bow_list[0].keys()
temp = {'change_id': ''}
temp.update(dict.fromkeys(keys, 0))
feature_list = []

# Iterate over each patch file and compute its BoW feature vector
path = './jdt/patch/*.patch'
files = glob.glob(path)

for name in files:
    try:
        allsentences = []
        bow_list = []
        with open(name, errors='ignore') as f:
            Lines = f.readlines()
            for line in Lines:
                # Only process deleted lines (lines starting with '-')
                if line.startswith('-'):
                    line = line[1:].strip()
                    # Skip diff headers, block comments, and single-line comments
                    if not (line.startswith('--') or line.startswith('*') or line.startswith('/*') or line.startswith('*/') or line.startswith('//')):
                        allsentences.append(remove_comments(line, '//'))

            # Generate BoW for the current patch file
            bow = generate_bow(allsentences)
            if bow is not None:
                bow_list.append(bow)

            # Create a feature row from the template and set the change_id
            template = temp.copy()
            patch_id = re.sub("[^0-9]", "", f.name.split('/')[3])
            template.update(change_id = patch_id)
                
            # Map BoW values into the template (only for keys in the global vocabulary)
            if len(bow_list) > 0:
                for key, value in bow_list[0].items():
                    if key in list(template.keys()):
                        template[key] = value

            feature_list.append(template)

    except IOError as exc:
        # Silently skip directories encountered during glob iteration
        if exc.errno != errno.EISDIR: 
            raise 

# =============================================================================
# Phase 3: Write the per-patch BoW features to a CSV file
# =============================================================================
with open('bow.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, temp.keys())
    dict_writer.writeheader()
    dict_writer.writerows(feature_list)
