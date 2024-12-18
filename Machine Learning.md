`                        `**Machine Learning**

The machine learning is a type of artificial intelligence that allows machine to learn and improve from the experience without being explicitly programme.

**Types of machine learning:**

1\. Supervised machine learning

2\. Un-supervised machine learning

3\. Reinforcement learning

**Supervised machine learning-**

**Linear Regression-**
**
`	`The linear regression is used to compute the linear relation between dependent and one or more independent features.

Simple linear regression – who having one independent value

Multi linear regression- who has more than one independent value

In the linear regression we are going to find the relation between independent variable and dependent variable. The best line should contain less error between the predicted and actual value.

To find out the best fit line we use :

`  `**y=mx+c** 

- Y is our output
- M is the slope 
- X is the input for the model
- C is the bias value

By the above equation we can find out the lines in the axis. Our aim is not to find out the line, finding out the best line which has less mean square error.

To achieve that we varying the m value and plotting the n number of lines in the graph. After plotting the planes we need to find the best fit line among them, for finding the best fit line we are going to use the gradient decent algorithm.

By the help of cost function we are going to draw a graph to finding out the best fit line. The gradient decent is brawn between cost function(CF) and m.

The convergence theorem find the global minimum in the graph and it was the best fit line for the model.

The model may be overfitting or underfitted we use ridge and lasso for removing the over and under fitting. 

**Logistic Regression-**

The logistic regression is used for binary classification. The logistic regression uses the sigmoid function to make predictions on the model.

The logistic regression model transforms linear regression function continuous values output into categorical output using the sigmoid function.

Sigmoid function:

**1/1+e^-y**

- Where y is the linear regression output y=mx+c
- In the logistic regression the ‘S’ shaped cure is drawn in the curve which divides the data points.
- The logistic regression gives the binary output 0 or 1, YES or NO,TRUE or FALSE.

The logistic regression is a ridge model.

After creating the s curve in the graph the model is trained using the traing data and predicting the output.

The process of forming a decision tree involves recursively partitioning the data based on the values of different attributes. The algorithm selects the best attribute to split the data at each internal node, based on certain criteria such as information gain or Gini impurity. This splitting process continues until a stopping criterion is met, such as reaching a maximum depth or having a minimum number of instances in a leaf node.



**Decision Tree**

`	`The decision tree is a supervised learning commonly used in machine learning and predict outcomes based on the data.

The decision tree is tree like structure it has a hierarical structure containing the root node, parent node, child node and leaf node. 

By the help of entropy and gini index we are going to split the data in the every feature. The root has the highest value means it is not pure split it is going to split until the split is pure.

To get the root node from the all features we are going to find the highest information gain among the features.

Which feature got the highest gain value it is taken as a root and starts splitting the data into a tree like structure 

Then train the model using the training data and gets the output. 

Hyper parameters:

1\.post pruning- post-pruning removes sections of the tree that provide little or no additional predictive power, improving the tree's ability to generalize to unseen data.  

2\.pre-pruning- max-depth,max-leaf(GridscanCV)

**Random Forest-**
**
`	`Random Forest algorithm is a powerful tree learning technique in Machine learning. It works by creating a number of Decision tree during the training phase. Each tree is constructed using a random subset of the data set to measure a random subset of features in each partition. This randomness introduces variability among individual trees, reducing the risk of overfitting and improving overall prediction performance.

The random forest uses the ensemble technique called bagging for the random forest.

Bagging- taking some amount of dataset and passes some amount of data for different models. Then they generated outputs and we are going to take the majority among them.

In  regression we are taking mean from the output.

The mean or majority taken from the model to predict the output.

The random forest takes the average of all the prediction made by the decision tree, which cancels out the bias, so it does not suffer from overfitting.   

**Support vector machine:**
**
`	`The support vector machine is a supervised machine learning. The primary objective of the SVM algorithm is to identify the optimal hyperplane in an N-dimensional space that can effectively separate data points into different classes in the feature space. The algorithm ensures that the margin between the closest points of different classes, known as support vectors, is maximized.

The svm creates a hyper plane with that additional it draws two margines. The svm aim to maximize the margin distance to get the accurate values.

The support vectors are the datapoints that passes through the margine plane in the svm graph. By using this we are creating the generalized model. The plane which has the large margine line that plane is taken as the  best fit line.

In the svm we are using some kernals that converts the low dimensional into higher dimension. The SVM kernel is a function that takes low-dimensional input space and transforms it into higher-dimensional space, ie it converts nonseparable problems to separable problems. It is mostly useful in non-linear separation problems. Simply put the kernel, does some extremely complex data transformations and then finds out the process to separate the data based on the labels or outputs defined.

Types of kernals:

1. Polynomial kernel
1. Gaussian RBF kernel
1. Sigmoid kernel
1. Linear kernel



**UNSUPERVISED MACHINE LEARNING:**

**K-means clustering:**

K-means clustering is a unsupervised machine learning algorithm used to group data points into clusters based on their similarities. The main idea behind K-means is to partition a dataset into K distinct clusters, where each cluster contains data points that are more similar to each other than to those in other clusters. The algorithm works iteratively to find the best way to group the data.

The process begins by selecting K initial centroids, which are the central points of the clusters. These centroids can be randomly chosen from the data points or set using other methods. Next, the algorithm assigns each data point to the nearest centroid, forming clusters. After all points are assigned, the centroids are recalculated as the mean of the points in each cluster. This assignment and updating process continues until the centroids no longer change significantly, indicating that the clusters have stabilized.

In the k means clustering the similar features are classified** into one cluster. The optimize clusters are taken from the  elbow method.

1\. We try the k-values

2\. Initialize k number of centroids

3\. Compute the average to update the centers

4\. compute the step 3 until we get the repeated values 

By using the elbow method we find out the optimized k values. By usind elbow method we draw a graph in the graph where we get the repeated values that’s the k value for the k-means clustering.

The goal of K-means clustering is to partition a set of data points into a specified number of distinct groups, called clusters, such that points within the same cluster are more similar to each other than to those in different clusters. By minimizing the variance within each cluster and maximizing the variance between clusters, K-means aims to identify natural groupings in the data. This helps in understanding patterns, organizing data effectively, and making it easier to analyze complex datasets by reducing their dimensionality and complexity.  

**Hierarchical clustering:**
**
`	`Hierarchical Clustering is an unsupervised machine learning algorithm used to group similar data points into clusters. Unlike other clustering methods like k-means, hierarchical clustering builds a tree-like structure of clusters, known as a dendrogram, which helps visualize how clusters are merged or split over different levels.

The dendrogram, a key visual tool in hierarchical clustering, provides a way to visualize the merging of clusters. On the x-axis, it shows individual data points or clusters, while the y-axis represents the distance or dissimilarity at which clusters are merged. The height of the branches in the dendrogram gives insight into how far apart clusters are when they are combined. By cutting the dendrogram at a specific height, users can decide the number of clusters.

Agglomerative hierarchical clustering follows a straightforward process. It starts by treating each data point as its own cluster, then calculates the pairwise distances between all points. The closest two clusters are merged based on the chosen linkage method. After merging, distances between the new cluster and the remaining clusters are recalculated, and this process is repeated until the desired number of clusters is formed or all points belong to a single cluster. This method doesn't require the user to specify the number of clusters in advance, which can be an advantage over algorithms like k-means. Instead, the dendrogram helps visually determine how many clusters exist by looking at significant gaps between merges.

To validate  the cluster model we use silihoutee score it ranges from -1 to +1.

The silihoutee score can be derived by a(i)=1/c1-1 derivation of I and j.

In the first cluster it checks and find the distance with the points that are present in the cluster and then taking the b(i) it checks the distance between one cluster to the another cluster.(b(i) > a(i)).

**Principal component analysis:**
**
`	`The pca is mainly used for dimensionality reduction of the dataframe.it is a unsupervised machine learning. The pca is used for feature extraction.

Principal Component Analysis (PCA) is a popular dimensionality reduction technique used in machine learning and data analysis to simplify complex datasets. By reducing the number of features, PCA makes it easier to analyze and visualize data, while preserving as much variability (information) as possible. 

The main goal of PCA is to simplify data, making it easier to analyze and visualize. By focusing on only the principal components that explain most of the variance, we can reduce the dataset’s dimensions without losing much information. Before applying PCA, the data is centered by subtracting the mean to ensure proper calculations. PCA is commonly used for data visualization, noise reduction, and as a preprocessing step in machine learning algorithms.

1. **Standardize the Data**: First, the data needs to be standardized by subtracting the mean of each feature and dividing by the standard deviation. This ensures that all features are on the same scale and centered around zero, which is crucial for PCA to work correctly.
1. **Calculate the Covariance Matrix**: After standardizing, calculate the **covariance matrix**, which shows how the different features are related to each other. The covariance matrix helps to understand the relationships between variables and their variance.
1. **Perform Eigen-Decomposition**: Next, the **eigenvectors** and **eigenvalues** of the covariance matrix are computed. The eigenvectors represent the directions of the new principal components, while the eigenvalues tell us how much variance is explained by each principal component.
1. **Sort the Principal Components**: The eigenvalues are sorted in descending order, and the eigenvectors are arranged accordingly. The principal component with the highest eigenvalue explains the most variance in the data and is chosen first, followed by the next most important component, and so on.
1. **Select the Number of Principal Components**: Decide how many principal components to keep based on the amount of variance they explain. Often, components that explain most of the variance (e.g., 90-95%) are retained, while the rest are discarded to reduce dimensionality.
1. **Transform the Data**: Finally, the original data is transformed into the new principal components by multiplying it with the eigenvectors. The result is a new set of features, which are uncorrelated and capture the most significant patterns in the data.

**Singular value decomposition**:

`	`Singular Value Decomposition (SVD) is a technique used in linear algebra and machine learning to factorize a matrix into three smaller matrices. It helps simplify complex data, making it easier to analyze and work with. In SVD, a matrix is broken down into three parts: **U**, Sigma, and **V^T**. The matrix **U** contains the left singular vectors, sigma is a diagonal matrix with singular values (representing the importance of each component), and **V^T** contains the right singular vectors.

SVD is often used for dimensionality reduction, similar to PCA, where it reduces the complexity of data by keeping only the most important components. It’s also used in applications like image compression, recommender systems, and latent semantic analysis for text data. By keeping only the largest singular values, we can simplify data without losing too much important information.

One of the strengths of SVD is that it can be applied to any kind of matrix, even non-square ones, making it very flexible. However, like PCA, it assumes that the most significant patterns in the data can be captured linearly. Despite this, SVD remains a powerful tool in machine learning and data science for simplifying data and extracting useful patterns.

**REINFORCEMENT LEEARNING:**

**Q-Learning:**
**
`	`Q-learning is a type of **reinforcement learning** used in machine learning to teach an agent how to make decisions in an environment. It helps the agent learn the best actions to take in different situations to maximize its rewards over time. The agent interacts with the environment, takes actions, and receives feedback in the form of rewards. Over time, it learns which actions lead to the highest rewards.

In Q-learning, the agent uses a **Q-table** to store information about the expected rewards for each action in each state of the environment. The goal is to update this table as the agent explores, so it can eventually know which action is best for each state. The formula used to update the Q-values balances immediate rewards with future rewards, which helps the agent learn long-term strategies, not just short-term gains.

Q-learning is useful for problems where an agent needs to make a series of decisions, like in games or robotic control tasks. It can take a long time to learn if the environment is large or complex.

**Deep Q-Network:**
**
`	`A Deep Q-Network (DQN) is a type of reinforcement learning method that uses deep learning to improve the decision-making process of an agent. Like traditional Q-learning, DQN helps an agent learn the best actions to take in different situations to maximize rewards over time. However, instead of using a Q-table, which becomes impractical for large environments, DQN uses a neural network to estimate Q-values.

The neural network takes in the current state of the environment as input and predicts the Q-values for all possible actions. The agent then selects actions based on these predicted values. Over time, the network learns to make better predictions by updating its weights through training, allowing the agent to get better at choosing the right actions.

DQN is especially powerful for environments with large or continuous state spaces, such as video games, where it's not feasible to store every possible state-action pair in a table. One of the key improvements of DQN is the use of experience replay, where the agent stores its experiences and learns from them in a random order, which stabilizes training. Another feature is the target network, which helps make the learning process more stable and efficient.





