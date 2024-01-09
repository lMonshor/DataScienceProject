import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import requests
from bs4 import BeautifulSoup
import pandas as pd


def extract_data_from_url():
    url = 'https://www.notebookcheck.net/Mobile-Processors-Benchmark-List.2436.0.html?type=&sort=&deskornote=4&gpubenchmarks=1&perfrating=1&or=0&showBars=1&3dmark06cpu=1&cpu_fullname=1&l2cache=1&l3cache=1&tdp=1&mhz=1&turbo_mhz=1&cores=1&threads=1&technology=1&daysold=1'
    html = requests.get(url).content
    soup = BeautifulSoup(html, 'html.parser')
    data = []
    rows = soup.find_all('tr')
    for row in rows:

        cols = row.find_all('td')
        cleaned_cols = []

        for col in cols:
            col_text = col.text.strip()
            cleaned_cols.append(col_text)

        data.append(cleaned_cols)

    return data


def scale_values(min_value, max_value, value):

    new_min = 1
    new_max = 1000

    scaled_value = round(
        ((value - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min)
    return scaled_value


def convert_to_kb(value):
    if 'MB' in value:
        mb_value = float(value.replace('MB', '').strip()) * \
            1024
        return int(mb_value)

    elif 'KB' in value:
        return int(value.replace('KB', '').strip())

    else:
        return 0


def categorize_manufacturer(model):
    if 'AMD' in model:
        return 'AMD'
    elif 'Intel' in model:
        return 'Intel'
    elif 'Qualcomm' in model:
        return 'Qualcomm'
    elif 'VIA' in model:
        return 'VIA'
    else:
        return 'Other'


def process_columns(df):

    df.columns = df.iloc[0]
    df = df[1:]

    df = df[df['Pos'] != 'Pos']

    df['Manufacturer'] = df['Model'].apply(categorize_manufacturer)

    df[['Base Clock(MHz)', 'Turbo Clock(MHz)']
       ] = df['MHz - Turbo'].str.split('‑', expand=True).apply(lambda x: x.str.strip())

    df[['Cores', 'Threads']
       ] = df['Cores / Threads'].str.split('/', expand=True)

    df[['L2 Cache(KB)', 'L3 Cache(KB)']
       ] = df['L2 Cache + L3 Cache'].str.split('+', expand=True)

    df['Process(nm)'] = df['Process (nm)']

    df['TDP(Watt)'] = df['TDP Watt'].str.split('.', expand=True)[0]

    df['Days Old'] = df['Days\xa0old']

    df['3DMark06(score)'] = df['3DMark06 CPU'].str.split(
        'n', expand=True)[0].str.split('.', expand=True)[0]

    df = df.drop(columns=['Pos', 'Perf. Rating', 'Cores / Threads', 'MHz - Turbo',
                 'L2 Cache + L3 Cache', 'Process (nm)', 'TDP Watt', 'Days\xa0old', '3DMark06 CPU'])

    df.fillna('0', inplace=True)

    df['L2 Cache(KB)'] = df['L2 Cache(KB)'].apply(convert_to_kb)
    df['L3 Cache(KB)'] = df['L3 Cache(KB)'].apply(convert_to_kb)

    column_names_except_model = df.columns[2:].tolist()

    for column in column_names_except_model:

        df[column] = pd.to_numeric(
            df[column], errors='coerce').astype('Int64')

    df.fillna(0, inplace=True)

    min_value = df['3DMark06(score)'].min()
    max_value = df['3DMark06(score)'].max()

    df['3DMark06(score)'] = df['3DMark06(score)'].apply(
        lambda x: scale_values(min_value, max_value, x))

    df.reset_index(drop=True, inplace=True)

    return df


raw_data = extract_data_from_url()

df = pd.DataFrame(raw_data)

df = process_columns(df)

#print(df)

df.to_csv('project/mining.csv')

print(df.info())



manufacturer_counts = df['Manufacturer'].value_counts()
plt.figure(figsize=(10, 6))
manufacturer_counts.plot(kind='bar')
plt.xlabel('Manufacturer')
plt.ylabel('Count')
plt.title('Number of Processors by Manufacturer')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8, 6))
plt.hist(df['3DMark06(score)'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('3DMark06 Score')
plt.ylabel('Frequency')
plt.title('Distribution of 3DMark06 Scores')
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
plt.boxplot(df['3DMark06(score)'].dropna(), vert=False)
plt.xlabel('3DMark06 Score')
plt.title('Distribution of 3DMark06 Scores')
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(df['Base Clock(MHz)'], df['3DMark06(score)'], alpha=0.7)
plt.xlabel('Base Clock (MHz)')
plt.ylabel('3DMark06 Score')
plt.title('Base Clock vs 3DMark06 Score')
plt.grid(True)
plt.show()


grouped = df.groupby(df['Days Old'] // 100 * 100)['3DMark06(score)'].mean()
plt.figure(figsize=(8, 6))
plt.plot(grouped.index, grouped.values, marker='o', linestyle='-')
plt.xlabel('Days Old (Grouped)')
plt.ylabel('Average 3DMark06 Score')
plt.title('Average 3DMark06 Score Grouped by Days Old')
plt.grid(True)
plt.show()




selected_columns = ['Base Clock(MHz)', 'Turbo Clock(MHz)', 'L2 Cache(KB)', 
    'L3 Cache(KB)', 'Cores', 'Threads', 'Process(nm)', 'TDP(Watt)', 'Days Old', '3DMark06(score)']
sns.heatmap(df[selected_columns].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Selected Features')
plt.show()



X = df.drop(['3DMark06(score)'], axis=1).select_dtypes(
    include=['int64'])
y = df['3DMark06(score)']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    evs = explained_variance_score(y_test, y_pred)

    print(f"{name}:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Explained Variance Score: {evs}")
    print("\n")
