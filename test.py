print("Execution started. \n Collecting the libraries required.")
import os
import warnings

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
#init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.plotly as py
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import plotly.offline as offline

# ignore warnings
warnings.filterwarnings("ignore")

print(os.listdir("../data visualization"))

world = pd.read_csv("../data visualization/countries of the world.csv")

print(world.head(10))

print(world.info())
#print(world.dtypes())


#Cleaning the data
world.columns = (["country","region","population","area","density","coastline","migration","infant_mortality","gdp","literacy","phones","arable","crops","other","climate","birthrate","deathrate","agriculture","industry","service"])

world.country = world.country.astype('category')
world.region = world.region.astype('category')
world.density = world.density.str.replace(",",".").astype(float)
world.coastline = world.coastline.str.replace(",",".").astype(float)
world.migration = world.migration.str.replace(",",".").astype(float)
world.infant_mortality = world.infant_mortality.str.replace(",",".").astype(float)
world.literacy = world.literacy.str.replace(",",".").astype(float)
world.phones = world.phones.str.replace(",",".").astype(float)
world.arable = world.arable.str.replace(",",".").astype(float)
world.crops = world.crops.str.replace(",",".").astype(float)
world.other = world.other.str.replace(",",".").astype(float)
world.climate = world.climate.str.replace(",",".").astype(float)
world.birthrate = world.birthrate.str.replace(",",".").astype(float)
world.deathrate = world.deathrate.str.replace(",",".").astype(float)
world.agriculture = world.agriculture.str.replace(",",".").astype(float)
world.industry = world.industry.str.replace(",",".").astype(float)
world.service = world.service.str.replace(",",".").astype(float)

print(world.info())

missing = world.isnull().sum()
print(missing)

world.fillna(world.mean(),inplace=True)
world.region = world.region.str.strip()

group = world.groupby("region")
group.mean()

print(world.head(10))

#Plotting the graph
region = world.region.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=region.index,y=region.values)
plt.xticks(rotation=45)
plt.ylabel('Number of countries')
plt.xlabel('Region')
plt.title('Number of Countries by REGİON',color = 'red',fontsize=20)
plt.show()


print(world.corr())

f,ax = plt.subplots(figsize=(18, 16))
sns.heatmap(world.corr(), annot=True, linewidths=.8, fmt= '.1f',ax=ax)
plt.show()

gdp=world.sort_values(["gdp"],ascending=False)
df = gdp.iloc[:100,:]

# Creating trace1
trace1 = go.Scatter(
                    x = df.gdp,
                    y = df.birthrate,
                    mode = "lines",
                    name = "Birthrate",
                    marker = dict(color = 'rgba(235,66,30, 0.8)'),
                    text= df.country)
# Creating trace2
trace2 = go.Scatter(
                    x = df.gdp,
                    y = df.deathrate,
                    mode = "lines+markers",
                    name = "Deathrate",
                    marker = dict(color = 'rgba(10,10,180, 0.8)'),
                    text= df.country)
z = [trace1, trace2]
layout = dict(title = 'Birthrate and Deathrate of World Countries (Top 100)',
              xaxis= dict(title= 'GDP',ticklen= 5,zeroline= False)
             )
fig = dict(data = z, layout = layout)
iplot(fig)#,image='png')
offline.plot(fig)


