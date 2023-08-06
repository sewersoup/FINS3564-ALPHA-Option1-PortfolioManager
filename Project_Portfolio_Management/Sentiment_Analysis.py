# Import libraries
import pandas as pd
# VADER sentiment analyser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import timedelta
import matplotlib.pyplot as plt

# Define constants
# Leverage pre-trained model due to limited data
nltk.download('vader_lexicon')
DEFAULTNEWSLOCATION = './news_dump.json'
FILELOCATION = './'
# Initialise vader sentiment analyser
SENTIMENTANALYZER = SentimentIntensityAnalyzer()


def loadData(newsLocation):
    """
    Given a *.json fiile location, loads Pandas df.
    Arg: newsLocation (string): Optional file location in json format.
    Returns: df (Pandas Dataframe)
    """
    if newsLocation == '':
        # If no location entered, provide the default
        newsLocation = DEFAULTNEWSLOCATION
    df = pd.read_json(newsLocation)
    # Convert date from string to datetime
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    return df


def scoreData(df, toScore):
    """
    Scores the sentiment of the data.
    Arg: df (Pandas Dataframe, optional file location in json format).
    Arg: toScore (Series): Equities to score
    Returns: meanScores (Pandas Dataframe, of sentiment analysis results).
    """
    # Use pretrained vader model to score
    # ! This is the biggest limitation of this model.
    # Ideally we would have more data to train our own model.
    scores = toScore.apply(SENTIMENTANALYZER.polarity_scores).tolist()
    #scores = list(toScore.apply(SENTIMENTANALYZER.polarity_scores))
    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)
    # Combine score with relevant input - this is for sanity checking
    df = df.join(scores_df, rsuffix='_right')
    print(df)
    df.to_csv('sentiment_scores.csv', index=False)

    # Group by equity, then find the mean sentiment over all occurances.
    # ! Ideally we weight more recent events.
    # Code is commented out as there are too few entries in the data given.
    """
    # This assumes that only last month contains relevent information.
    # Filter out all other info.
    mostRecentDate = df['Date/Time'].max()
    df = df[(df['Date/Time'] > (mostRecentDate - timedelta(days=30)))]
    df.to_csv(FILELOCATION + 'reduced_sentiment_scores.csv')
    """
    meanScores = df.groupby(['Equity']).mean()
    print('Sentiment analysis results:')
    print(meanScores)
    return meanScores


def graphSentiment(meanScores):
    """
    Given the dataframe and the mean scores, the function plots the average sentiment
    score of each Equity.
    Arg: meanScores (Pandas DataFrame)
    """
    plt.figure(figsize=(10, 5))
    plt.bar(meanScores.index, meanScores['compound'])
    plt.xlabel('Equity')
    plt.ylabel('Average Sentiment Score')
    plt.title('Average Sentiment Score per Equity')
    plt.xticks(rotation=45)
    plt.show()


def analyseNews(newsLocation):
    """
    Given news, returns a dataframe estimating the impact of news on each stock.
    Arg: newLocation (json doc - provide filepath to json news).
    """
    # Stations 1 & 2: Extract, Transform and Load (ETL) & Feature Engineering.
    df = loadData(newsLocation)
    # Station 3 - Model Design
    meanScores = scoreData(df, df['Headline'])
    # Station 4 - Implementation
    graphSentiment(meanScores)
    return (meanScores)


def equitiesSentiment(newsLocation):
    """
    Given location of json news file, the function returns the
    adjustments to make for each stock due to sentiment analysis.
    Arg: newsLocation (str - location of news file).
    Returns: equityWeights (DataFrame - adjustment to make for each
    stock, limited to +/-5%.
    """
    equityCompound = analyseNews(newsLocation)
    # Find compound adjustment and transpose so equities are columns.
    return equityCompound[['compound']].T

print(equitiesSentiment('news_dump.json'))
