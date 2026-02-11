import sys
import glob
import errno
import re
from sklearn.feature_extraction.text import CountVectorizer
import csv

def word_extraction(sentence):    
    keywords = ['abstract', 'assert', 'boolean', 'byte', 'catch', 'char', 'class', 'const', 'double', 'enum', 'exports', 'extends', 'final', 'finally', 'float', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', 'package', 'private', 'protected', 'public', 'short', 'static', 'super', 'this', 'throw', 'throws', 'try', 'void', 'util', 'java']    

    words = re.split(r"\s+|\.|\(|\)|\{|\}|\[|\]|\;|\=|\!|\&|\||\+|\-|\*|\%|\>|\<|\?|\:|\"|\#|\'|\,|\^|\\n|\\t|\d+", sentence) 
    cleaned_text = [w for w in words if w not in keywords and w != '']    
    return cleaned_text

def tokenize(sentences):    
    words = []    
    for sentence in sentences:        
        w = word_extraction(sentence)        
        words.extend(w)            
        
    return words

def generate_bow(allsentences):        
    vocab = tokenize(allsentences)
    if vocab:
        vectorizer = CountVectorizer()
        vectorizer.fit_transform(vocab).todense()
        keyword_dict = vectorizer.vocabulary_
        for key in [key for key in keyword_dict if keyword_dict[key] < 4]: del keyword_dict[key] 
        
        return keyword_dict

def remove_comments(line, sep):
    for s in sep:
        i = line.find(s)
        if i >= 0:
            line = line[:i]
    return line.strip()


allsentences = []
bow_list = []
path = './jdt/patch/*.patch'   
files = glob.glob(path)   
for name in files: # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
    try:
        with open(name, errors='ignore') as f: # No need to specify 'r': this is the default.
            Lines = f.readlines()
            for line in Lines:
                if line.startswith('-'):
                    line = line[1:].strip()
                    if not (line.startswith('--') or line.startswith('*') or line.startswith('/*') or line.startswith('*/') or line.startswith('//')):
                        allsentences.append(remove_comments(line, '//'))
            
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise


bow = generate_bow(allsentences)

if bow is not None: # and len(bow) > 3:
    bow_list.append(bow)


keys = bow_list[0].keys()
temp = {'change_id': ''}
temp.update(dict.fromkeys(keys, 0))
feature_list = []

# iterate the files and collect bow for each files
path = './jdt/patch/*.patch'
files = glob.glob(path)
for name in files: # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
    try:
        allsentences = []
        bow_list = []
        with open(name, errors='ignore') as f: # No need to specify 'r': this is the default.
            Lines = f.readlines()
            for line in Lines:
                if line.startswith('-'):
                    line = line[1:].strip()
                    if not (line.startswith('--') or line.startswith('*') or line.startswith('/*') or line.startswith('*/') or line.startswith('//')):
                        allsentences.append(remove_comments(line, '//'))

            bow = generate_bow(allsentences)
            if bow is not None: # and len(bow) > 3:
                bow_list.append(bow)

            template = temp.copy()
            patch_id = re.sub("[^0-9]", "", f.name.split('/')[3])
            template.update(change_id = patch_id)
                
            if len(bow_list) > 0:
                for key, value in bow_list[0].items():
                    if key in list(template.keys()):
                        template[key] = value

            feature_list.append(template)

    except IOError as exc:
        if exc.errno != errno.EISDIR: 
            raise 


with open('bow.csv', 'w') as output_file:
    #for elem in feature_list:
    dict_writer = csv.DictWriter(output_file, temp.keys())
    dict_writer.writeheader()
    dict_writer.writerows(feature_list)
