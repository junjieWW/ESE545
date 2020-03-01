# LSH(Locality Sensitive Hashing) alogrithm
In computer science, locality-sensitive hashing (LSH) is an algorithmic technique that hashes similar input items into the same "buckets" with high probability. (The number of buckets are much smaller than the universe of possible input items.)[1] Since similar items end up in the same buckets, this technique can be used for data clustering and nearest neighbor search. It differs from conventional hashing techniques in that hash collisions are maximized, not minimized. Alternatively, the technique can be seen as a way to reduce the dimensionality of high-dimensional data; high-dimensional input items can be reduced to low-dimensional versions while preserving relative distances between items.


## 1. Preprocessing of the dataset

The dataset of users and their reviews is loaded using pandas. 
There are mainly three steps of preprocessing: removing punctuations, removing stop words, convert the text to lower case. For example, the sentence ‘Great!!!!’ will be transformed to ‘great’.

## 2. Represent the documents in term of binary matrix

To compare each review, we need to transfer them into a set of k-shingles. After that, I create a dictionary for distinct shingles and use index to record them. To build the binary matrix, we just need to traverse each document (a set of k-shingles) and try to look up if they are in the dictionary. The shape of binary matrix should be (length of dictionary, number of documents), since I try to use each column to represent one document. For each document, if it contains shingles that appear in the dictionary, the value should be set to 1. Otherwise, it should be 0 in the document column. 

In my project, I choose k = 5, which means the length of each shingle should be 5. For those shingles that contain less than 5 characters, I use space to pad them. The number of k chosen here is according to the hint. If k is too small, say 1, then every document will contain exactly the same thing, which means every review is close to each other. If k is too charge, it is very slow to construct the binary matrix and it is possible that every review would be exactly the same to be the close neighbor. So, I choose k = 5.


## 3. Distributions of dataset
To get a sense for the data, let’s peform a random sample. The histogram of the pairwise Jaccard distances is shown below, it can be seen the most of pairs are far from each other. The average distance of random 10,000 pairs is 0.987 and the minimum distance of random 10,000 pairs is 0.91. The distance function we used here is Jaccard distance, it is calculated based using 1 – intersection of two documents/union of two documents. 
 

## 4. Efficient storage of data
As is discussed above, it is a waste of memory to use a huge binary matrix to store the shingles. Here, I use a data structure ‘list of list’ to store the information of shingles. Since we only cares about positions of 1 in the binary matrix, we can only store the index where 1 appears. So the final data structure is in such form: there are 157683 lists in a list. In each list, I only store the index in each row of the binary matrix.
## 5. Detect all pairs that are close to one another
This is the core part of this entire project. It is computationally cost to compare every pairs in 157683 documents, the complexity of the calculation would be O(N2). Thus, we need to find a way to speed up computation. One way of doing this is to use locally sensitive hashing. The basic idea of it is to hash each document to a bucket and only consider documents that are in the same bucket are close neighbors. Ideally, if we want to reduce the ‘misses’, we can apply multiple independently random hash functions. I define the permutation function as (a*x + b) % prime.  The parameter a and b is chosen randomly, and their sizes are of (M,1). The prime number should be larger than the length of the dictionary (row number of binary matrix). M is the number of permutation functions. Here, I choose M = 100 to construct the signature matrix. Increasing the number of hash functions can reduces the false-negative rate, but also there will be an increasing number of false-positive rate. After tuning parameters. I choose M = 100 to construct the signature matrix.

After constructing the signature matrix, we can compare get a size of (M, num_doc). Now, we could use this matrix to determine which documents are fallen into the same bucket. However, we could also optimize our algorithms by hashing the signature matrix. By choosing the number of bands b, and the number of row in each band r, we can hashing the signature matrix. The parameter I determine is dependent on the threshold. Since we want the distance of documents in one bucket to be less than 0.2, we can plot the graph of Pr(hit)-sim(C1,C2) to find the suitable parameter.
 
The similarity we want is 0.8. In the graph above, it can be seen that the appropriate number of bands should be 10. In such case, the number of false positives and false negatives are relatively small. After implementing the algorithm, I find 6901 pairs that are close to each other. The sample is like following graph:
 
 
## 6 Complexity of implementation
Naïve implementation, after we construct the signature matrix, the cost of computation is still O(N2).  Now our implementation break down to O(N), we only have to compute the documents in the same bucket, which is really a small range. The original comparison is around O(150k2). Now, it is down to O (76013* (1~3)). In this representation, 76013 means number of buckets, and 1~3 means number of reviews in the bucket. Most buckets contain only two documents.




