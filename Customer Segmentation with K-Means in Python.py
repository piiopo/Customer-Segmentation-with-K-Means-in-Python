#IMPORTING PACKAGES

import pandas as pd # working with data frame
import numpy as pd # working with arrays
import matplotlib.pyplot as plt # visualization 
import seaborn as sb # visualization
from mpl_toolkits.mplot3d import Axes3D #3d plot
from termcolor import colored as cl # text customization

from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.cluster import KMeans # K-means algorithm

plt.rcParams['figure.figsize'] = (20, 10)
sb.set_style('whitegrid')

# IMPORTING DATA

import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CSV/customer.csv')
df.drop('Unnamed: 0', axis = 1, inplace = True)
df.set_index('Customer Id', inplace = True)

df.head()

# Age distribution

print(cl(df['Age'].describe(), attrs = ['bold']))

sb.distplot(df['Age'], 
            color = 'salmon')
plt.title('AGE DISTRIBUTION', 
          fontsize = 18)
plt.xlabel('Age', 
           fontsize = 16)
plt.ylabel('Frequency', 
           fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

plt.savefig('age_distribution.png')
plt.show()

# Age distribution

print(cl(df['Age'].describe(), attrs = ['bold']))

sb.distplot(df['Age'], 
            color = 'salmon')
plt.title('AGE DISTRIBUTION', 
          fontsize = 18)
plt.xlabel('Age', 
           fontsize = 16)
plt.ylabel('Frequency', 
           fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

plt.savefig('age_distribution.png')
plt.show()

#Ave vs Income

sb.scatterplot('Age','Income',
               data = df,
               color = 'deepskyblue',
               s = 300,
               alpha = 0.6,
               edgecolor = 'b')

plt.title('AGE / INCOME',
          fontsize = 18)

plt.xlabel('Age',
           fontsize = 16)

plt.ylabel('Income',
           fontsize = 16)

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

plt.savefig('age_income.png')
plt.show()


# Years Employed vs Income

area = df.DebtIncomeRatio **2

#While building the Years Employed/Income graph. 
#Got an error, Defaulted records length was 700 instead of 850 due to NA records in the csv file. Solved it using:
df['Defaulted'] = df['Defaulted'].fillna(0)

sb.scatterplot('Years Employed', 'Income', 
               data = df, 
               s = area, 
               hue = 'Defaulted',
               alpha = 0.6, 
               edgecolor = 'white')


plt.title('YEARS EMPLOYED / INCOME', 
          fontsize = 18)
plt.xlabel('Years Employed', 
           fontsize = 16)
plt.ylabel('Income', 
           fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

plt.legend(loc = 'upper left', fontsize = 14)

plt.savefig('y_income.png')
plt.show()      


# DATA PROCESSING
import numpy as np

X = df.values
X = np.nan_to_num(X)

sc = StandardScaler()

cluster_date = sc.fit_transform(X)

print(cl('Cluster data samples : ', attrs= ['bold']), cluster_date[:5])

cluster = 3
model = KMeans(init = 'k-means++',
               n_clusters = cluster,
               n_init = 12)

model.fit(X)

labels = model.labels_

print(cl(labels[:100], attrs = ['bold']))

df['cluster_num'] = labels
df.head()

df.groupby('cluster_num').mean()


area = np.pi * (df.Edu) ** 4

sb.scatterplot('Age', 'Income', 
               data = df, 
               s = area, 
               hue = 'cluster_num', 
               palette = 'spring', 
               alpha = 0.6, 
               edgecolor = 'darkgrey')
plt.title('AGE / INCOME (CLUSTERED)', 
          fontsize = 18)
plt.xlabel('Age', 
           fontsize = 16)
plt.ylabel('Income', 
           fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(loc = 'upper left', fontsize = 14)

plt.savefig('c_age_income.png')
plt.show()



fig = plt.figure(1)
plt.clf()
ax = Axes3D(fig, 
            rect = [0, 0, .95, 1], 
            elev = 48, 
            azim = 134)

plt.cla()
ax.scatter(df['Edu'], df['Age'], df['Income'], 
           c = df['cluster_num'], 
           s = 200, 
           cmap = 'spring', 
           alpha = 0.5, 
           edgecolor = 'darkgrey')
ax.set_xlabel('Education', 
              fontsize = 16)
ax.set_ylabel('Age', 
              fontsize = 16)
ax.set_zlabel('Income', 
              fontsize = 16)

plt.savefig('3d_plot.png')
plt.show()