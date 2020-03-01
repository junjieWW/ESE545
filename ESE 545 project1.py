#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections


# In[2]:


data = pd.read_json(r'amazonReviews.json', lines=True)[['reviewerID', 'reviewText']]

def remove_stopwords(text):
    clean_words = []
    word_list = text.split()
    stop_words = ["i", "me", "my", "myself", 
     "we", "our", "ours", "ourselves",
     "you", "your", "yours", "yourself",
     "yourselves", "he", "him", "his", 
     "himself", "she", "her", "hers", "herself",
     "it", "its", "itself", "they", "them",
     "their", "theirs", "themselves", "what",
     "which", "who", "whom", "this", "that", 
     "these", "those", "am", "is", "are", "was",
     "were", "be", "been", "being", "have", "has", 
     "had", "having", "do", "does", "did", "doing", 
     "a", "an", "the", "and", "but", "if", "or", 
     "because", "as", "until", "while", "of", "at", 
     "by", "for", "with", "about", "against", "between",
     "into", "through", "during", "before", "after", 
     "above", "below", "to", "from", "up", "down", 
     "in", "out", "on", "off", "over", "under", "again",
     "further", "then", "once", "here", "there", "when", 
     "where", "why", "how", "all", "any", "both", "each",
     "few", "more", "most", "other", "some", "such", "no",
     "nor", "not", "only", "own", "same", "so", "than", "too", 
     "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    
    for word in word_list:
        if word not in stop_words:
            clean_words.append(word)
    
    
    return ' '.join(clean_words)
    


# In[36]:


def preprocess(text):
    # lower case
    text = text.lower()
    
    # remove puntuations
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
      
    for x in text: 
        if x in punctuations: 
            text = text.replace(x, "") 
    
    # remove stop words
    text = remove_stopwords(text)
    
    return text 
    


# In[37]:


def k_shingle(k, text):
    
    hashset = set()
    
    if len(text) >= k:
        for pos, char in enumerate(text):
            if pos + k <= len(text):
                hashset.add(text[pos:pos+k])
    else:
        hashset.add(text + ' '* (k - len(text)))
        
    return hashset


# In[38]:


def get_prepared(raw_text):
    prepare = []
    for single_text in raw_text:
        text = preprocess(single_text)
        text = k_shingle(5,text)

        prepare.append(text)
    return prepare


# In[39]:


def build_dictionary(prepare):
    dic = {}
    index = 0
    for doc_set in prepare:
        for val in doc_set:
            if val not in dic:
                dic[val] = index
                index += 1
    
    return dic


# In[40]:


def build_binary_matrix(dic,prepare):
    
    matrix = np.zeros((len(dic),len(prepare)))
    
    for col, doc in enumerate(prepare):
        for shingle in doc:
            if shingle in dic:
                matrix[dic[shingle],col] = 1
    
    return matrix
    # store it in the sparse matrix


# In[41]:


def build_dense_matrix(dic, prepare):
    sparse_list = []
    for doc in prepare:
        indices = []
        for shingle in doc:
            if shingle in dic:
                indices.append(dic[shingle])
        sparse_list.append(indices)
    return sparse_list
                


# # Problem 2

# In[42]:


def jaccard_distance(doc1,doc2):
    unions = len(set(doc1 + doc2))
    intersections = len(set(doc1).intersection(doc2))
        
    return 1 - intersections/unions


# In[43]:


def cal_avg_jaccard(matrix):
    print('start to random choose 10,000 pairs and compute their distance...')
    total_num = len(matrix)
    total_pairs = 10000
    
    pairs = np.random.choice(total_num,(total_pairs,2),replace = False)
    distance = []
    for pair in pairs:
        
        doc1 = matrix[pair[0]]
        doc2 = matrix[pair[1]]

        d = jaccard_distance(doc1,doc2)
        
        distance.append(d)
        
        
    avg_distance = sum(distance)/ total_pairs
    min_distance = min(distance)
    
    print('Finished! The following is the distribution of distance.')
    
    print('The average distance of 10000 pairs is : ', avg_distance)
    print('The minimum distance of 10000 pairs is : ', min_distance)
    
    plt.hist(distance,100)
    plt.title('Histogram for distance of random 10,000 pairs')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.show()


# # Problem 3

# In[44]:


def find_next_prime(n):


    for p in range(n, 2*n):
        for i in range(2, p):
            if p % i == 0:
                break
        else:
            return p
    return None


# In[45]:


def generate_signature_matrix(data_list,dic, M):

 
    prime = find_next_prime(len(dic))
    a = np.random.choice(prime,M,replace = False).reshape(M,1)
    b = np.random.choice(prime,M,replace = False).reshape(M,1)


    # Generating signature matrix
    signature_matrix = np.zeros((M, len(data_list)))

    for col in range(signature_matrix.shape[1]):
        signature_matrix[:,col] = ((a*np.array(data_list[col]) + b)% prime).min(1)

    return signature_matrix


# In[46]:


def get_candidate_pairs(signature,dic,num_bands):
    
    # need to pick number of bands b and number of rows
    # pick 10 bands and 10 rows
    M = signature.shape[0]
    num_doc = signature.shape[1]
    
    num_rows = int(M/num_bands) # number of rows in each band
    
    # first random choose parameters for permutation
    prime = find_next_prime(len(dic))
    a = np.random.choice(prime,num_rows,replace = False).reshape(num_rows,1)
    b = np.random.choice(prime,num_rows,replace = False).reshape(num_rows,1)
    # i is the number of total bands
    hash_table = np.zeros((num_bands, num_doc))
    for i in range(num_bands):
        band_matrix = signature[i*num_rows:(i+1)*num_rows, :]
        hash_table[i,:] = np.sum((a * band_matrix + b)%prime, axis = 0)
    return hash_table   
         


# In[47]:


def find_pairs(hash_table):
    results = []
    for i in range(len(hash_table)):
        counter=collections.Counter(hash_table[i])
        distinct_hash_values = [i for i in counter if counter[i] > 1] 
        for value in distinct_hash_values:
            results.append(np.where(hash_table[i] == value)[0])
    return results
    


# In[86]:


def get_pairs(result,data_list,raw_text):
    distance_list = []
    for pairs in result:
        for i in range(len(pairs)-1):
            for j in range(i+1, len(pairs)):
                distance = jaccard_distance(data_list[pairs[i]], data_list[pairs[j]])
                if distance < 0.2:
                    distance_list.append((pairs[i],pairs[j],raw_text[pairs[i]], raw_text[pairs[j]]))
    return distance_list


# In[65]:

print('start preprocessing the dataset....')
text_list = data['reviewText'].tolist()
text_list[:] = [x for x in text_list if x] 
prepare = get_prepared(text_list)
dic = build_dictionary(prepare)
data_list = build_dense_matrix(dic, prepare)
print('Finished! Now the dataset is stored as binary matrix')
# In[66]:


cal_avg_jaccard(data_list)


# In[80]:
print('start generating signature matrix...')
signature = generate_signature_matrix(data_list,dic,100)
print('signature matrix generated!')

print('start getting similar pairs...')
hash_table = get_candidate_pairs(signature,dic,10)

results = find_pairs(hash_table)

candidates = get_pairs(results,data_list,text_list)
print('already get the pairs!')
import csv

with open('closeneighbors.csv','w') as writeFile:
    file = csv.writer(writeFile, delimiter = ',')
    for i in range(len(candidates)):
        file.writerow([candidates[i][0],candidates[i][1],candidates[i][2],candidates[i][3]])

print('complete writing pairs to the file, total number of is',len(candidates))
# In[49]:


def find_closest_neighbour(review,data):
    getID = {}
    count = 0 
    
    for idx,text in enumerate(data['reviewText']):
        key = data['reviewerID'][idx]
        if text:
            getID[count] = key
            count += 1
        
    raw_text = data['reviewText'].tolist()
    raw_text[:] = [x for x in raw_text if x] 
    raw_text.append(review)
    new_prepare = get_prepared(raw_text)
    new_dic = build_dictionary(new_prepare)
    new_data_list = build_dense_matrix(new_dic, new_prepare)
    new_signature = generate_signature_matrix(new_data_list,new_dic,100)
    new_hash_table = get_candidate_pairs(new_signature,new_dic,10)
    query_review_value = new_hash_table[:,-1]
    buckets = []
    for row,value in enumerate(query_review_value):
        buckets.append(np.where(new_hash_table[row] == value)[0])
    
    new_doc_index = len(raw_text)-1
    min_distance = 1
    reviewID = 0
    for pairs in buckets:
        for i in range(len(pairs)-1):
            distance = jaccard_distance(new_data_list[pairs[i]],new_data_list[new_doc_index])
            if distance < min_distance:
                reviewID = pairs[i]
    
    return getID[reviewID], raw_text[reviewID]    
          


# In[91]:


def tuning_parameters():
    Ms = [100,500,1000]
    s = np.arange(0,1,0.001)
    for m in Ms:
        r = [2,4,5,10,20,50]
        b = [m/i for i in r]
        for i in range(len(r)):        
            plt.plot(s,1-(1-s**r[i])**b[i])
        plt.vlines(0.8, 0, 1, colors = "c", linestyles = "dashed")
        plt.legend(labels = r, loc = 'best')
        plt.xlabel('Sim')
        plt.ylabel('Pr(hit)')
        plt.show()


review = input("Please put a review here: ")
review_ID, content = find_closest_neighbour(review, data)
print('The most closet review is: '+content+ ' by ' + review_ID)

