# Product-Recommender
A Python(TensorFlow) based ML model that recommends products based on user purchase history and rating history.
To run:
1) Run the command "pip install -are Requirements.txt" to install dependencies.
2) Then run the command "streamlit run Frontend.py".(Note: For Windows users, this command needs to be executed in an anaconda terminal)
3) Note: First-time run will cause the program to do some pre-calculations required to run the program.(This will take some time depending on your computer)
4) Add products using the "Add Product" button (You can add multiple products), and then click on the "Give Recommendation" button to get recommendations.

What does it do:
1) The Model takes the dataset of all products with at least 5 reviews from the dataset: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/.
2) The model then classifies the products using "tfidfvectorizer()" to get a sparse matrix, which contains the value of relations of all products to unique keywords found among all products.
3) Then we take the user product history and get the relation between all the products and keywords we already had. Then we use the sigmoid_kernel method to take the vector sigmoid of the matrix with the original trained matrix containing the entire product dataset.
4) Then we take the mean of the sigmoid values pertaining to each original product and sort the result in Descending order to get our recommendations in terms of Content Filtering.
5) Then it takes the dataset of other users and their ratings and forms a matrix of correlations using the 'pearson' method.
6) After which it will take the user history and first check if the product is present in the dataset of users, if not then we take the most similar item based on content filtering to replace it.
7) Then for each product we find the item-item to collaborative filter recommendation, sum these values and apply sigmoid to it to get our final distribution of recommendations.
8) Then we take the mean of the content filtering and collaborative filtering approach and sort the sigmoid values one last time to get out final recommendations.
