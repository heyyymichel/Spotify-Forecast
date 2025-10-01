import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

dataset1 = pd.read_csv("datasets\high_popularity_spotify_data.csv") # first half of dataset
dataset2 = pd.read_csv("datasets\low_popularity_spotify_data.csv") # second half of dataset

dataset = pd.concat([dataset1,dataset2],ignore_index=True) #combine datasets

dataset = dataset.drop_duplicates() #drop duplicates after combining

dataset['release_date'] = pd.to_datetime(dataset['track_album_release_date'],errors='coerce') #convert dates

dataset['release_year'] = dataset['release_date'].dt.year #extract release year

dataset['genre'] = dataset['playlist_genre'].str.lower().str.strip() #fix text inconsistencies

dataset = dataset.dropna(subset=['track_popularity']) #drop nulls in dataset

dataset['era'] = dataset['release_year'].apply(lambda x: 'Before 2020' if x<2020 else '2020+')

dataset = dataset.drop_duplicates(subset=['track_name'])

#code below plots genre popularity before & after a specific year
plt.figure(figsize=(12,6))
sns.boxplot(data=dataset[dataset['genre'].isin(['pop','rock','hip-hop'])],x='genre',y='track_popularity',hue='era')
plt.title("Genre Popularity Before and After 2020")
plt.show()

#code below plots growth of genres over time
genre_trend = dataset.groupby(['release_year','genre'])['track_popularity'].mean().reset_index()
plt.figure(figsize=(14,6))
popular_genres = ['pop','rock','hip-hop','afrobeats','r&b','latin','brazilian','gospel']
sns.lineplot(data=genre_trend[genre_trend['genre'].isin(popular_genres)],x='release_year',y='track_popularity',hue='genre')
plt.title("Genre Popularity Trends Over Time")
plt.show()

#export data for dataset
dataset.to_csv("datasets\cleaned_combined_spotify_data.csv",index=False)

#Group average popularity by genre and year
genre_yearly = dataset.groupby(['release_year','genre'])['track_popularity'].mean().reset_index()
#store results
results = []

#combines all actual vs prediced into one file
allData = []

#Forecast for top 5 genres
for g in popular_genres:
    genres = genre_yearly[genre_yearly['genre']== g][['release_year','track_popularity']]
    genres = genres.rename(columns={'release_year':'ds','track_popularity':'y'})
    genres['ds'] = pd.to_datetime(genres['ds'],format='%Y')

    #skip genres with little data
    if genres.shape[0] < 6:
        print()
        continue
    
    try:
        model = Prophet()
        model.fit(genres)

        #Predict next 10 years
        future = model.make_future_dataframe(periods=10,freq='YE')
        #Generate forecast
        forecast = model.predict(future)

        #Normalize dates to year-start for clean merge
        genres['ds'] = genres['ds'].dt.to_period('Y').dt.to_timestamp()
        forecast['ds'] = forecast['ds'].dt.to_period('Y').dt.to_timestamp()

        #prepare actual vs predicted for Power BI export
        actual = genres[['ds','y']].copy()
        actual['genre'] = g
        actual['type'] = 'Actual'
        actual = actual.rename(columns={'y': 'popularity'})

        predicted = forecast[['ds','yhat']].copy()
        predicted['genre'] = g
        predicted['type'] = 'Predicted'
        predicted = predicted.rename(columns={'yhat': 'popularity'})

        combined = pd.concat([actual,predicted],ignore_index=True)
        allData.append(combined)
        combined.to_csv(f"datasets\{g}_actual_vs_predicted.csv",index=False)

        #code below calculates accuracy metrics
        genre_cv = cross_validation(
            model,
            initial='1095 days', #inital = amount of historical data used to train model, this case 5 years
            period='365 days', #Re-train and test every year
            horizon = '730 days', #Forecast for 2 years
        )

        metrics = performance_metrics(genre_cv)
        avgMetrics = metrics.mean()
        results.append({
            'genre':g,
            'mae':round(avgMetrics['mae'],2),
            'rmse':round(avgMetrics['rmse'],2),
            'mape':round(avgMetrics['mape'],2),
            'coverage':round(avgMetrics['coverage'],2)
        })
        
        #Plot forecast
        model.plot(forecast)
        plt.title(f"Forecasted Popularity of {g}")
        plt.xlabel("Year")
        plt.ylabel("Predicted Popularity")
        plt.tight_layout()
        plt.show()

        #Plot actual vs predicted
        plt.figure(figsize=(10,6))
        plt.plot(actual['ds'],actual['popularity'],label='Actual',marker='o')
        plt.plot(predicted['ds'],predicted['popularity'],label='Predicted',marker='x')
        plt.title(f"Actual vs Predicted Popularity for {g}")
        plt.xlabel("Year")
        plt.ylabel("Track Popularity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error processing {g}: {e}")
        continue



all_results = pd.concat(allData,ignore_index=True)
all_results.to_csv("datasets\\all_genres_actual_vs_predicted.csv",index=False)

accuracy = pd.DataFrame(results)
print("\n Forecast Accuracy Summary:")
print(accuracy)
accuracy.to_csv("datasets\genre_forecast_accuracy.csv",index=False)



