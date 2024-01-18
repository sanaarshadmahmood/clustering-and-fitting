import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


def get_data():
    """
    This function read data from csv file using the pandas library,
    transpose the dataframe columns and clean the datagframe and
    return the dataframe object.
    :return: data
    """
    file_path = 'API_NY.GDP.PCAP.CD_DS2_en_csv_v2_6298251.csv'
    data = pd.read_csv(file_path, skiprows=4)

    non_year_columns = ['Country Code', 'Indicator Name', 'Indicator Code']
    data = data.drop(columns=non_year_columns, errors='ignore')
    data = data.set_index('Country Name')
    data = data.replace('..', np.nan).dropna(axis=1, how='all').astype(float)

    data = data.ffill(axis=1).bfill(axis=1)
    data = data.fillna(data.mean())
    return data


def linear_model(x, a, b):
    """
    This function take three input parameters and calculate the fit
    value using the linear model.
    model
    :param x:
    :param a:
    :param b:
    :return: calculated_value
    """
    calculated_value = a * x + b
    return calculated_value


sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (15, 8)})

# call function to read dataset and get dataframe
data = get_data()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.T).T

# create cluster
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10).fit(scaled_data)
data['Cluster'] = kmeans.labels_

plt.figure(figsize=(18, 10))

for i in range(4):
    cluster_data = data[data['Cluster'] == i]
    for index, row in cluster_data.iterrows():
        # make line plot and add label
        plt.plot(row[:-1], label=f'{index} (Cluster {i})')

# set plot title, axis labels and legend
plt.title('GDP per Capita Clustering', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('GDP per Capita (Normalized)', fontsize=14)
plt.legend()
plt.show()

us_data = data.loc['United States', data.columns[:-1]].dropna()
years = np.array([int(year) for year in us_data.index])
gdp_values = us_data.values

# calculate fit value using linear model
popt, pcov = curve_fit(linear_model, years, gdp_values)
predicted_gdp = linear_model(years, *popt)

# create line plots for GPD and data fitting using linear model
plt.figure(figsize=(15, 8))
plt.plot(years, gdp_values, label='Actual GDP', color='blue', marker='o')
plt.plot(years, predicted_gdp, label='Fitted Linear Model',
         linestyle='--', color='red')

# set plot title, axis labels and legend
plt.title('Linear Model Fitting for United States GDP per Capita',
          fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('GDP per Capita', fontsize=14)
plt.legend()
plt.show()

for i in range(4):
    plt.figure(figsize=(15, 8))
    cluster_data = data[data['Cluster'] == i]
    for index, row in cluster_data.iterrows():
        plt.plot(row[data.columns[:-1]], label=index)
    plt.title(f'GDP per Capita Trends in Cluster {i}', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('GDP per Capita', fontsize=14)
    plt.legend()
    plt.show()

X, y = make_blobs(n_samples=200, centers=5, cluster_std=0.60, random_state=42)

spectral_model_rbf = SpectralClustering(n_clusters=5,
                                        affinity='nearest_neighbors')
labels_spectral = spectral_model_rbf.fit_predict(X)

plt.figure(figsize=(12, 6))

# make subplot and show the spectral clustering results using scatter plot
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels_spectral, cmap='viridis', marker='.')
plt.title('Spectral clustering results')

# make subplot and show the ground truth clustering results using scatter plot
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='.')
plt.title('Ground truth clustering')

plt.tight_layout()
plt.show()
