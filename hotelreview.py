# Import the pandas library.
import pandas
import matplotlib.pyplot as plt
from time import time
# Import the kmeans clustering model.
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
# Import the linearregression model.
from sklearn.linear_model import LinearRegression


print ("Reading in the data.")
review = pandas.read_csv("winemag-data-130k-v2.csv")
# Print the names of the columns in games.
print(review.columns)
print(review.shape)
#print(review.head(10))
print(review.dtypes)

#Basic Graph plot
#plt.hist(review["points"])
#plt.show()


#Clean the Dataset
review.columns = ["ID","country","description","designation","points","price","province","region1","region2","tastername","tastertwitterhandle","title","variety","winery"]
review.ID = review.ID.astype('category')
review.country = review.country.astype('category')
review.description = review.description.astype('category')
review.designation = review.designation.astype('category')
review.points = review.points.astype(float)
review.price = review.price.astype(float)



print(review.dtypes)
print(review.isnull().sum())
review = review.drop('designation', 1)
review = review.drop('region1', 1)
review = review.drop('region2', 1)
review = review.drop('tastername', 1)
review = review.drop('tastertwitterhandle', 1)
review = review.drop('description', 1)
review = review.drop('title', 1)
review = review.drop('variety', 1)
review = review.drop('winery', 1)
review = review.drop('province', 1)
review = review.drop('ID', 1)

print(review.isnull().sum())
print("Started Cleaning")
t0 = time()
review.fillna(review.mean(),inplace=True)
clean_time = time() - t0
print("finished Cleaning in %fseconds"%clean_time)
print(review.isnull().sum())
print(review.corr())


# Initialize the model with 2 parameters -- number of clusters and random state.
kmeans_model = KMeans(n_clusters=3, random_state=1)
# Get only the numeric columns from games.
good_columns = review._get_numeric_data()
# Fit the model using the good columns.
kmeans_model.fit(good_columns)
# Get the cluster assignments.
labels = kmeans_model.labels_


# Create a PCA model.
pca_2 = PCA(2)
# Fit the PCA model on the numeric columns from earlier.
plot_columns = pca_2.fit_transform(good_columns)
# Make a scatter plot of each game, shaded according to cluster assignment.
#plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
#plt.title("Clustering of Vine Price Vs Rating ")
# Show the plot.
#plt.show()

# Generate the training set.  Set random_state to be able to replicate results.
train = review.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = review.loc[~review.index.isin(train.index)]
# Print the shapes of both sets.
print(train.shape)
print(test.shape)

# Initialize the model class.
model = LinearRegression()
# Fit the model to the training data.
model.fit(train[good_columns.columns.tolist()], train["points"])


# Import the scikit-learn function to compute error.
from sklearn.metrics import mean_squared_error

# Generate our predictions for the test set.
predictions = model.predict(test[good_columns.columns.tolist()])

# Compute error between our test predictions and the actual values.
mean_squared_error(predictions, test["points"])


print("Finished")




