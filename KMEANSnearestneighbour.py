'''
  The Hello World of the kneighbours algorithm with supervised learning.
  Plottig it in graph to manually verify it.
  Change the colour of the nearest neighbour
'''
import numpy as np
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plot

x = np.array([[-2, 5],[2,5],[0,0],[3,3],[5,2],[-4,4],[0,3],[1,3],[-1,3],[2,4],[0,1],[3,2]])
y= np.array([["true1"],["false1"],["true1"],["true1"],["false1"],["true1"],["true1"],["true1"],["false1"],["true1"],["true1"],["false1"]])
#print(x)


nb = NearestNeighbors(n_neighbors=1,algorithm="ball_tree").fit(x,y)
#nb = NearestNeighbors(n_neighbors=2).fit(x)
distance,indices = nb.kneighbors(x)
print(indices)
#print(distance)
predictValue = np.array([[2,3]])
predictDist,predictIndex =nb.kneighbors(predictValue)
print(predictIndex)

temp = predictIndex[0].tolist()
i = -1
for m in indices.tolist():
    i= i+1
    if(temp == m):
        print("bingo")
        nearestnode = x[[i]]
        print( y[[i]] )
        #plot.plot(nearestnode[:,0],nearestnode[:,1],'o',c='green')
        #plot.show()
        #break

#print(predictDist)
plot.plot(x[:,0],x[:,1],'o')
plot.plot(predictValue[:,0],predictValue[:,1],'o',c='red')
plot.plot(nearestnode[:,0],nearestnode[:,1],'o',c='green')
plot.show()

print(nb.kneighbors_graph(x).toarray())

