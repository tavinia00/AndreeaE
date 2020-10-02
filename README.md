Project: Identify customer segments - uses unsupervised machine learning in the organization of large and complex datasets

- In this project, I use two datasets one with demographic information about the people of Germany, 
and one with that same information for customers of a mail-order sales company.
I look at relationships between demographics features, organize the population into clusters, and see how prevalent customers are in each of the segments obtained.

Step 1: Preprocessing
I explore and understand the data by assesing how are missing or unknown values encoded in the data, if there are certain features (columns) that should be removed from the
analysis because of missing data, if there are certain data points (rows) that should be treated separately from the rest. I consider the level of measurement for each feature
in the dataset (e.g. categorical, ordinal, numeric). What assumptions must be made in order to use each feature in the final analysis? Are there features that need to be re-encoded before they can be used? Are there additional features that can be dropped at this stage?
I will create a cleaning procedure that you will apply first to the general demographic data, then later to the customers data.

Step 2: Feature Transformation
Now that data is clean, I will use dimensionality reduction techniques to identify relationships between variables in the dataset, resulting in the creation of a new set of 
variables that account for those correlations. In this stage of the project, I will attend to the following points:

The first technique that I perform on the data is feature scaling. What might happen if we don’t perform feature scaling before applying later techniques you’ll be using?
Once I’ve scaled the features, I can then apply principal component analysis (PCA) to find the vectors of maximal variability. How much variability in the data does each principal
component capture? Can I interpret associations between original features in your dataset based on the weights given on the strongest components? How many components will I
keep as part of the dimensionality reduction process?
I will use the sklearn library to create objects that implement your feature scaling and PCA dimensionality reduction decisions.

Step 3: Clustering
Finally, on the transformed data, I will apply clustering techniques to identify groups in the general demographic data. 
I will then apply the same clustering model to the customers dataset to see how market segments differ between the general population and the mail-order sales company. 
I will tackle the following points in this stage:
Use the k-means method to cluster the demographic data into groups. How should I make a decision on how many clusters to use?
Apply the techniques and models that I fit on the demographic data to the customers data: data cleaning, feature scaling, PCA, and k-means clustering.
Compare the distribution of people by cluster for the customer data to that of the general population. 
Can I say anything about which types of people are likely consumers for the mail-order sales company?
sklearn will continue to be used in this part of the project, to perform your k-means clustering. 
In the end, I will export the completed notebook with your work as an HTML file, which will serve as a report documenting your approach and findings.

