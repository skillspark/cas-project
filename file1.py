# %% [markdown]
# # CAS BDAI Individual Innovation Project: Tennis Match Predictor
# 
# ## Table of Contents 
# 1. [Introduction](#introduction)
# 2. [Preliminary steps](#preliminary-steps)
# 3. [ ](# )
# 4. [ ](# )
# 5. [ ](# )
# 6. [ ](# )
# 7. [ ](# )
# 8. [ ](# )
# 

# %% [markdown]
# ## Introduction <a name="introduction"></a>
# 
# ### Tennis Match Predictor: GAImeSetMatch
# 
# ### Goal of this project
# 
# ### Steps to implement
# 1. Load and explore the data
# 2. Data processing and cleaning
# 3. Feature Engineering
#     - Surface win %
#     - Tournament level win %
#     - Head-to-head
#     - Recent form
# 4. Data Analysis
# 5. Prediction
# 
# 
# ![.png](img/project/image.png)
# 
# Image source: [something](https://example.com/)

# %% [markdown]
# ## Preliminary steps <a name="preliminary-steps"></a>

# %% [markdown]
# ### Set the path to the interpreter (OPTIONAL - skip if using Google Colab; modify if using local dev environment )

# %%
#!/home/jean/Documents/dev/cas-project/venv_proj/bin/python3

# %% [markdown]
# ### Import the dependencies
# We need to import the required libraries.

# %%
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import re
# %matplotlib inline

# %% [markdown]
# ### Set static parameters
# Here we set some parameters which won't be changed. This allows for more easy handling and viewing of the data being explored.

# %%
# first, set some static parameters and options (used later too for loading other files)

# directory containing the .csv files
DIRNAME = 'data'

# set options for pandas viewing
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# pd.reset_option('display.float_format')

# %% [markdown]
# ### Define some helper functions and datasets
# These will help us later with common tasks.

# %% [markdown]
# #### Sample data 5 observations

# %%

# Small sample dataframe (5 matches) for misc usage
sample_matches_df = pd.DataFrame(data = {
    'tourney_id': ['2000-301', '2000-301', '2000-301', '2000-301', '2000-301'],
    'tourney_name': ['Auckland', 'Auckland', 'Auckland', 'Auckland', 'Auckland'],
    'surface': ['Hard', 'Hard', 'Hard', 'Hard', 'Hard'],
    'draw_size': [32, 32, 32, 32, 32],
    'tourney_level': ['A', 'A', 'A', 'A', 'A'],
    'tourney_date': [20000110, 20000110, 20000110, 20000110, 20000110],
    'match_num': [1, 2, 3, 4, 5],
    'winner_id': [103163, 102607, 103252, 103507, 102103],
    'winner_seed': [1.0, None, None, 7.0, None],
    'winner_entry': [None, 'Q', None, None, 'Q'],
    'winner_name': ['Tommy Haas', 'Juan Balcells', 'Alberto Martin', 'Juan Carlos Ferrero', 'Michael Sell'],
    'winner_hand': ['R', 'R', 'R', 'R', 'R'],
    'winner_ht': [188.0, 190.0, 175.0, 183.0, 180.0],
    'winner_ioc': ['GER', 'ESP', 'ESP', 'ESP', 'USA'],
    'winner_age': [21.7, 24.5, 21.3, 19.9, 27.3],
    'loser_id': [101543, 102644, 102238, 103819, 102765],
    'loser_seed': [None, None, None, None, 4.0],
    'loser_entry': [None, None, None, None, None],
    'loser_name': ['Jeff Tarango', 'Franco Squillari', 'Alberto Berasategui', 'Roger Federer', 'Nicolas Escude'],
    'loser_hand': ['L', 'L', 'L', 'L', 'L'],
    'loser_ht': [180.0, 183.0, 173.0, 185.0, 185.0],
    'loser_ioc': ['USA', 'ARG', 'ESP', 'SUI', 'FRA'],
    'loser_age': [31.1, 24.3, 26.5, 18.4, 23.7],
    'score': ['7-5 4-6 7-5', '7-5 7-5', '6-3 6-1', '6-4 6-4', '0-6 7-6(7) 6-1'],
    'best_of': [3, 3, 3, 3, 3],
    'round': ['R32', 'R32', 'R32', 'R32', 'R32'],
    'minutes': [108.0, 85.0, 56.0, 68.0, 115.0],
    'w_ace': [18.0, 5.0, 0.0, 5.0, 1.0],
    'w_df': [4.0, 3.0, 0.0, 1.0, 2.0],
    'w_svpt': [96.0, 76.0, 55.0, 53.0, 98.0],
    'w_1stIn': [49.0, 52.0, 35.0, 28.0, 66.0],
    'w_1stWon': [39.0, 39.0, 25.0, 26.0, 39.0],
    'w_2ndWon': [28.0, 13.0, 12.0, 15.0, 14.0],
    'w_SvGms': [17.0, 12.0, 8.0, 10.0, 13.0],
    'w_bpSaved': [3.0, 5.0, 1.0, 0.0, 6.0],
    'w_bpFaced': [5.0, 6.0, 1.0, 0.0, 8.0],
    'l_ace': [7.0, 10.0, 6.0, 11.0, 8.0],
    'l_df': [8.0, 7.0, 6.0, 2.0, 8.0],
    'l_svpt': [106.0, 74.0, 56.0, 70.0, 92.0],
    'l_1stIn': [55.0, 32.0, 33.0, 43.0, 46.0],
    'l_1stWon': [39.0, 25.0, 20.0, 29.0, 34.0],
    'l_2ndWon': [29.0, 18.0, 7.0, 14.0, 18.0],
    'l_SvGms': [17.0, 12.0, 8.0, 10.0, 12.0],
    'l_bpSaved': [4.0, 3.0, 7.0, 6.0, 5.0],
    'l_bpFaced': [7.0, 6.0, 11.0, 8.0, 9.0],
    'winner_rank': [11.0, 211.0, 48.0, 45.0, 167.0],
    'winner_rank_points': [1612.0, 157.0, 726.0, 768.0, 219.0],
    'loser_rank': [63.0, 49.0, 59.0, 61.0, 34.0],
    'loser_rank_points': [595.0, 723.0, 649.0, 616.0, 873.0]
}
)

# %% [markdown]
# ## Load and explore the data
# This section loads the data available in .csv files from the aforementioned source, explores the data and then cleans it for ease of use and data quality.

# %% [markdown]
# 
# ### Load matches
# Data is available in the form of results of ATP matches. For simplicity reasons, focus only on matches since the year 2000*. Each year is stored in one file using naming convention atp_matches_yyyy.csv.
# 
# *The reasoning behind this: since the year 2000, there have been factors that have influenced the outcomes of the modern form of the sport. For me, these are:
# 1. Racquet technology: Since the 1980s, rackets are made mainly out of graphite. Reference: [Link](https://www.pledgesports.org/2019/08/evolution-of-tennis-rackets/)
# 2. String technology: In the late 1990s, polyester strings were introduced, which revolutionised the sport. Reference: [Link](https://scientificinquirer.com/2021/08/30/string-theory-the-synthetic-revolution-that-changed-tennis-forever/)
# 3. Surfaces: in 2009, the ATP discontinued use of carpet court use in all its tournaments. Reference: [Link](https://racketsportsworld.com/tennis-not-played-carpet-courts/#When_was_Carpet_Discontinued_from_Use_in_Tennis)

# %%
# create a list of matches (since the year 2000 ) files to load
atp_match_files = [f'{DIRNAME}/atp_matches_{year}.csv' for year in range(2000, 2024)]

# %%
# create an empty dataframe to store all matches
matches_df = pd.DataFrame()

# loop through the list of match files, read them and append the data to the combined DataFrame
for filen in atp_match_files:
    matches_df = pd.concat([matches_df, pd.read_csv(filen, index_col=None)])


# %%
# explore the matches data
matches_df.head()

# %%
# get an overview of number of features, instances, empty values and data types 
matches_df.info()

# %% [markdown]
# Alll features starting with "w_" or "l_" indicate in-game metrics, which is out of scope for this project. So we will remove them later. 

# %%
matches_df.describe()

# %%
print("Amount of instances and features: " + str(matches_df.shape))

# %% [markdown]
# ### Exploring the matches

# %% [markdown]
# #### Zeros
# Here we check for zeros in the matches mframe, in order to decide what to do with them.

# %%
# check all features for zero's
zero_count_per_feature= matches_df.apply(lambda col: (col == 0).sum())
zero_count_per_feature

# %%
# explore the matches with 0 or less minutes
matches_lessthan_0mins = matches_df.loc[matches_df['minutes']<=0]
matches_lessthan_0mins.head()

# %% [markdown]
# The matches lasting 0 minutes are all W/O ("Walkovers"), meaning that one player did not contest the match due to injury, illness, etc. These instances should not be used for predicting matches, as they don't measure a player's performance. 

# %% [markdown]
# #### Score contains text
# Sometimes the score feature contains text, like "RET" (match retirement), in addition to the previously observation about W/O. If we want to calculate the number of games played, we should remove this later.

# %%
matches_score_text = matches_df[matches_df['score'].str.contains('[a-zA-Z]')]
matches_score_text.head()

# %% [markdown]
# #### NaN or empty values
# Here we check for NaN or empty values in the matches mframe, in order to decide what to do with them.

# %%
# check all features for empty values
empty_count_per_feature= matches_df.isnull().sum()
empty_count_per_feature

# %% [markdown]
# Besides the features starting with "w_" or "l_", there are 8 features in the matches dataset which have empty values, and indication whether this will be used for prediction or not:
# 1. minutes - not used
# 2. seed - not used
# 3. entry - not used
# 4. hand - not used
# 5. ht (height) - not used
# 6. age - not used
# 7. rank - used
# 8. rank_points - not used
# 
# Of these 8 features, only 1 will be used: rank. Let's explore a few of these matches with an empty rank. 

# %%
# explore the matches with empty rank
matches_empty_rank = matches_df.loc[matches_df['winner_rank'].isnull() | matches_df['loser_rank'].isnull()]
matches_empty_rank.head()

# %% [markdown]
# The matches with players having no (empty) rank could be because they are new, or have been inactive due to injury and hence lost their ranking before returning. We can try and look up their last valid ranking in the rankings file later. 
# 
# Next, do the values for rank make sense?

# %%
matches_with_rank = matches_df.loc[~matches_df['winner_rank'].isnull() & ~matches_df['loser_rank'].isnull()]

# Plot 2 histograms for distribution of values for "rank"
# Create subplots for the histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot the first histogram for winner_rank
ax1.hist(matches_with_rank['winner_rank'], bins=20, color='blue', alpha=0.7)
ax1.set_title('Distribution of winner rank')
ax1.set_xlabel('Rank')
ax1.set_ylabel('Frequency')

# Plot the second histogram for loser_rank
ax2.hist(matches_with_rank['loser_rank'], bins=20, color='green', alpha=0.7)
ax2.set_title('Distribution of loser rank')
ax2.set_xlabel('Rank')
ax2.set_ylabel('Frequency')

# Display the histograms
plt.tight_layout()
plt.show()

# %% [markdown]
# The histogram shows that most matches are won by players ranked in the top 100 (~60'000), which makes sense. Also, there are no outlier values like rank=5'000.

# %%
# could matches with empty minutes be due to the tourney_level?
# print(matches_empty_minutes['tourney_level'].value_counts())

# %% [markdown]
# #### Tournament start dates
# It would be interesting to see on which weekdays tournaments start. Becuase later, we want to link the rankings data with the matches data, so a common day of the week  would be required.

# %%
# convert date from tourney_date 
matches_tournament_starts = matches_df.copy()
matches_tournament_starts['tourney_date_dt'] = pd.to_datetime(matches_df['tourney_date'], format='%Y%m%d')

# create a column representing the day of the week
matches_tournament_starts['tourney_date_dt_day_name'] = matches_tournament_starts['tourney_date_dt'].dt.day_name()

# day of week frequency for matches and rankingsday of week frequency for matches and rankings
matches_tournament_starts['tourney_date_dt_day_name'].value_counts(normalize=True)

# %% [markdown]
# As seen above, ca. **81%** of the matches started on a Monday. This is a strong case to say that for simplicity, we set all matches to start at the beginning of the week which would be Monday. But before doing this, let's see which matches don't start on a Monday and group by tournament type, then display the results using a bar chart.

# %%
# Group the matches_tournament_starts by 'tourney_level' and 'tourney_date_dt_day_name' and count the occurrences
matches_tournament_starts = matches_tournament_starts.groupby(['tourney_level', 'tourney_date_dt_day_name']).size().unstack().fillna(0)

# Create a stacked bar chart
matches_tournament_starts.plot(kind='bar', stacked=True, figsize=(10, 6))

# Set labels and title
plt.xlabel('Tournament Type')
plt.ylabel('Count')
plt.title('Tournament Start Weekday by Tournament Type')

# Display the legend
plt.legend(title='Tournament Start Weekday', loc='upper right')

# Show the plot
plt.show()

# %% [markdown]
# From [matches_data_dictionary.txt](data/matches_data_dictionary.txt):
# - 'G' = Grand Slams
# - 'M' = Masters 1000s
# - 'A' = other tour-level events
# - 'C' = Challengers
# - 'S' = Satellites/ITFs
# - 'F' = Tour finals and other season-ending events
# - 'D' = Davis Cup 
# 
# Most tournaments start on a Monday, with a notable exception: Davis Cup, which are run over weekends and start on a Friday. 
# **Decision**: For better linking with rankings, we've decided that we will set all tournaments' start dates to the Monday which precedes it. For example, if its Friday yyyy-mm-dd, then a supplemental date feature will be provided for its preceding Monday yyyy-mm-dd.

# %% [markdown]
# ### Prediction benchmark for matches
# In order to evaluate the prediction accuracy of our model, we need a benchmark to compare when predicting the results of matches. One simple benchmark would be to assume that the higher (i.e. closer to 1) ranked player will always win. This "higher-ranked player win ratio" can easily be calculated using the features available in the original dataset.
# We know that some rankings are empty, so we will just substitute a number higher than the max. ranking (which is 2101).

# %%
# setting a prediction benchmark, empty ranking means no ranking, so replace with a arbitrary high value
matches_wins_by_ranking_df = matches_df.copy()
matches_wins_by_ranking_df[['winner_rank','loser_rank']] = matches_wins_by_ranking_df[['winner_rank','loser_rank']].fillna(value=10000)

# add a new feature which is the result of checking whether the winner was ranked higher (i.e. closer to 1) than the loser
matches_wins_by_ranking_df['winning_player_ranked'] = matches_wins_by_ranking_df.apply(lambda x: "higher" if x['winner_rank'] < x['loser_rank'] else "lower", axis=1)
matches_wins_by_ranking_df['winning_player_ranked'].value_counts(normalize=True)*100

# %% [markdown]
# So we see that for our dataset, the higher ranked player won **65.6%** of all the matches. This will be our benchmark for evaluating the model.

# %% [markdown]
# ### Load rankings
# Data is also available in the form of ranking of ATP players. It may be required to supplement the missing data for current rankings in the matches dataset, for example, a player doesn't have a ranking at the time of playing a match. 

# %% [markdown]
# ### Exploring the rankings data

# %%
# create a list of rankings (since the year 2000 ) files to load
atp_rankings_files = [f'{DIRNAME}/atp_rankings_{year}.csv' for year in ['00s','10s', '20s', 'current']]

# %%
# create an empty dataframe to store all rankings
rankings_df = pd.DataFrame()

# loop through the list of rankings files, read them and append the data to the combined DataFrame
for filen in atp_rankings_files:
    rankings_df = pd.concat([rankings_df, pd.read_csv(filen, index_col=None)])


# %%
# explore the rankings data
rankings_df.head()

# %%
# get an overview of number of features, instances, empty values and data types 
rankings_df.info()

# %%
# sanity checks on the data (min values, max values, etc.)
rankings_df.describe()

# %% [markdown]
# From the above table, the min and max values for the rankings make sense. Also, the ranking_date makes sense. Finally, there are no missing values, so no data cleaning is required on this dataset.

# %%
print("Amount of instances and features: " + str(rankings_df.shape))

# %% [markdown]
# #### Ranking dates
# Similar to which weekdays tournaments start, let's look at the days on which the rankings get updated. Becuase later, we want to link the rankings data with the matches data, so a common day of the week  would be required.

# %%
# convert date from ranking_date 
ranking_update = rankings_df.copy()
ranking_update['ranking_date_dt'] = pd.to_datetime(rankings_df['ranking_date'], format='%Y%m%d')

# create a column representing the day of the week
ranking_update['ranking_date_dt_day_name'] = ranking_update['ranking_date_dt'].dt.day_name()

# day of week frequency for ranking
ranking_update['ranking_date_dt_day_name'].value_counts(normalize=True)

# %% [markdown]
# **All (100%)** of the rankings are updated on a Monday. Therefore, we are aligned with the idea to set all tournament start dates to a Monday.
# 
# Below is a final view of the loaded data for rankings, with the new column for the datetime formatted `ranking_date_dt`

# %%
# check data types
print(rankings_df.info())

# preview data
rankings_df.head()

# %% [markdown]
# ## Data processing and cleaning
# 
# ### Scope of processing and cleaning
# 1. Replace the matches' winner and loser columns
# 2. Clean the date features and make them consistent
# 3. Ensure the matches are sorted as needed
# 4. Remove matches with result as W/O
# 5. Players without rankings: 
# - seasoned players  (they had a long layoff due to injury, etc.). keep match and lookup ranking from earlier. Apply penalty of 10 ranking places for each week they were absent.
# - if they played less than 10 matches (cumulative) - remove match
# 

# %% [markdown]
# ### Execute the processing and cleaning

# %% [markdown]
# #### Start with a copy of the original loaded dataframes

# %%
matches_processed_df = matches_df.copy()
rankings_processed_df = rankings_df.copy()

# %% [markdown]
# #### Hide winner and loser from columns names
# Replace columns starting with 'winner_' and 'loser_' with 'player_1_' and 'player_2_' for the required features. As we want to be able to predict who will be the winner and the loser in each match, we remove the 'winner_' and 'loser_' columns for each match, and instead replace it with player_1_ and player_2 according to which the ranking of the players. 
# 
# The features starting with 'w_' and 'l_' are measures recorded during the match and will not be used in the model for predicting the outcome, so we remove these features.
# We will add a column at the end of the dataframe, which will serve as our y variable.

# %%
def hide_winner_loser(input_df):
    # List of required features to be replaced with prefixes player_1 and player_2
    features = ['id', 'seed', 'entry', 'name', 'hand', 'ht', 'ioc', 'age', 'rank', 'rank_points']
    
    # Copy the input DataFrame to a new one
    df = input_df.copy()

    # Add player_1_name and player_2_name columns based on higher rank
    df['player_1_name'] = np.where((df['winner_rank'].fillna(float('inf')) <= df['loser_rank'].fillna(float('inf'))),
                                   df['winner_name'],
                                   df['loser_name']
                                   )
    df['player_2_name'] = np.where((df['winner_rank'].fillna(float('inf')) > df['loser_rank'].fillna(float('inf'))),
                                   df['winner_name'],
                                   df['loser_name']
                                   )

    # Transfer the values from 'winner_' and 'loser_' features to 'player_1_' and 'player_2_' features, according to who was the winner & loser
    for feat in features:
        player_1_feature = np.where(df['player_1_name'] == df['winner_name'],
                                    df['winner_' + feat],
                                    df['loser_' + feat]
                                    )
        player_2_feature = np.where(df['player_2_name'] == df['winner_name'],
                                    df['winner_' + feat],
                                    df['loser_' + feat]
                                    )
        df['player_1_' + feat] = player_1_feature
        df['player_2_' + feat] = player_2_feature   

          
    # Add a winner column
    df['winner'] = df.apply(lambda row: 'player_1' if row['winner_name'] == row['player_1_name'] else 'player_2', axis=1)

    # Remove columns starting with 'winner_' and 'loser_' (they have been replaced by player_1_ and player_2_)
    df = df.loc[:, ~df.columns.str.startswith('winner_') & ~df.columns.str.startswith('loser_')]

    # Remove columns starting with 'w_' and 'l_' (not needed for predicting_)
    df = df.loc[:, ~df.columns.str.startswith('w_') & ~df.columns.str.startswith('l_')]

    return df


# %% [markdown]
# Let us test the function hide_winner_loser with a sample dataset of 5 instances. Observe the renamed features, from "winnner_" and "loser_" to "player_1" and " player_2", and the new feature called "winner" (our y variable).

# %%
output_df = hide_winner_loser(sample_matches_df)
output_df.info()

# %%
output_df[['tourney_id'
           , 'player_1_name', 'player_1_rank'
           , 'player_2_name', 'player_2_rank']]

# %%
# replace the winner and loser columns with player_1 and player_2 for the matches dataset
matches_processed_df= hide_winner_loser(matches_processed_df)
matches_processed_df.head()

# %% [markdown]
# #### Clean and consistent date features

# %%
# create new column for datetime datatype version of the date columns
matches_processed_df['tourney_date_dt'] = pd.to_datetime(matches_processed_df['tourney_date'], format='%Y%m%d')
rankings_processed_df['ranking_date_dt'] = pd.to_datetime(rankings_processed_df['ranking_date'], format='%Y%m%d')

# %% [markdown]
# #### Ensure the matches are sorted as needed
# This is crucial as we are calculating cumulative measures (e.g. count of prior matches) to base a prediction on. It's not required for the rankings dataset.

# %%
# sort matches by tourney_date, tourney_id and match_num, and reset the index as the old one is not required anymore.
matches_processed_df = matches_processed_df.sort_values(['tourney_date', 'tourney_id', 'match_num'], ascending=True)
matches_processed_df = matches_processed_df.reset_index(drop=True) 
matches_processed_df.head()

# %% [markdown]
# #### Remove matches with result as W/O
# W/O stands for "Walkover". Matches resulting in W/O should not be considered, so remove them. 

# %%
# remove matches resulting in a W/O
matches_processed_df = matches_processed_df[matches_processed_df['score'] != 'W/O']
len(matches_processed_df)

# %% [markdown]
# Down from 71'213 to 70'910 instances

# %% [markdown]
# #### Remove matches with 0 or less minutes
# Not needed

# %%
# remove matches with 0 or less minutes
# matches_df = matches_df.loc[matches_df['minutes']<0]


# %% [markdown]
# #### Add feature for matches dataset that all tournaments start dates are shown as a Monday

# %%
# add feature for tournaments not starting on a Monday, with its value being the preceding Monday
matches_processed_df['tourney_date_dt'] = pd.to_datetime(matches_processed_df['tourney_date'], format='%Y%m%d')
matches_processed_df['tourney_date_dt_preceding_monday'] = matches_processed_df['tourney_date_dt'].apply(lambda x: x - pd.DateOffset(days=x.weekday()) if x.weekday() != 0 else x)

# verify that this feature's date values are all on a Monday
matches_processed_df['tourney_date_dt_preceding_monday'].dt.day_name().value_counts()

# %%
# check examples of these new column values compared to its original
matches_processed_df[matches_processed_df['tourney_date_dt'].dt.day_name() != 'Monday'][['tourney_date_dt', 'tourney_date_dt_preceding_monday']].groupby(['tourney_date_dt', 'tourney_date_dt_preceding_monday']).size().reset_index(name='count').head(3)

# %% [markdown]
# - 28 Jan. 2000 was a Friday, and 24 Jan. 2000 was the preceding Monday
# - 4 Feb. 2000 was a Friday, and 31 Jan. 2000 was the preceding Monday
# - 17 Mar. 2000 was a Friday, and 13 Mar. 2000 was the preceding Monday

# %%
# matches_processed_df.to_csv("matches_processed_df.csv", sep=",", header=True)

# %% [markdown]
# #### Process matches with new players having no ranking 
# Remove matches where 1 opponent has so far played less than 10 completed matches. 
# Notes: 
# - Don't remove matches in the year 2000, as our players could have played 10 matches prior to the year 2000, and our cumulative count features need a year to get working.
# - W/O matches don't count, but retirements do.

# %%
### example of players where 1 opponent has so far played < 10 completed matches

# %% [markdown]
# #### Process matches with seasoned players having no ranking
# As explained before, there are matches with seasoned (experienced on the ATP Tour) players having no (empty) rank possibly because they have been inactive due to injury and hence lost their ranking before returning. If they are not new players, we can try and look up their last valid ranking in the rankings file. A recent example is Kevin Anderson, who was inactive for a period due to retiring in May 2022 and then announcing his comeback in July 2023* 
# 
# *Source: [Wikipedia "Kevin_Anderson (tennis)", accessed Oct. 2023](https://en.wikipedia.org/wiki/Kevin_Anderson_(tennis))
# 
# We will: 
# 1. for a particular match, find the latest available historical ranking in the rankings dataset for the player in the matches dataset
# 2. add 10 to the ranking for each week where the player was inactive.

# %%
# example of player previously having a ranking but later no ranking
matches_processed_ka_df = matches_processed_df[(matches_processed_df['player_1_name'] == 'Kevin Anderson') 
                                                | (matches_processed_df['player_2_name'] == 'Kevin Anderson')].tail()
matches_processed_ka_df[['tourney_date_dt', 'player_1_id', 'player_2_id', 'player_1_name', 'player_2_name', 'player_1_rank', 'player_2_rank']]

# %%
# define static parameter
rank_penalty_per_week_inactivity = 10

# define a function to look up historic rankings for players having no ranking in a particular match
def impute_missing_rankings(m, r):
    last_rankings = {}  # Dictionary to store the last available player_rank for each player_id

    for i, row in m.iterrows():
        if pd.isna(row['player_1_rank']):
            week = row['tourney_date_dt'] - dt.timedelta(days=row['tourney_date_dt'].weekday())
            p1_id = row['player_1_id']

            # Find the last available ranking date prior to the tourney_date_dt
            last_ranking_date = r[(r['player'] == p1_id) & (r['ranking_date_dt'] < week)]['ranking_date_dt'].max()

            if last_ranking_date:
                last_ranking_row = r[(r['player'] == p1_id) & (r['ranking_date_dt'] == last_ranking_date)]
                if not last_ranking_row.empty:
                    last_rank = last_ranking_row['rank'].values[0]
                    weeks_difference = (week - last_ranking_date).days // 7
                    imputed_rank = last_rank + weeks_difference * rank_penalty_per_week_inactivity
                    if imputed_rank > 3333: # Don't over-penalize
                        m.at[i, 'player_1_rank'] = 3333   # Set a default value   
                    else: 
                        m.at[i, 'player_1_rank'] = imputed_rank
                        last_rankings[p1_id] = imputed_rank
                else:
                    m.at[i, 'player_1_rank'] = 3333  # Set a default value
            else:
                m.at[i, 'player_1_rank'] = 3333  # Set a default value

        if pd.isna(row['player_2_rank']):
            week = row['tourney_date_dt'] - dt.timedelta(days=row['tourney_date_dt'].weekday())
            p2_id = row['player_2_id']

            # Find the last available ranking date prior to the tourney_date_dt
            last_ranking_date = r[(r['player'] == p2_id) & (r['ranking_date_dt'] < week)]['ranking_date_dt'].max()

            if last_ranking_date:
                last_ranking_row = r[(r['player'] == p2_id) & (r['ranking_date_dt'] == last_ranking_date)]
                if not last_ranking_row.empty:
                    last_rank = last_ranking_row['rank'].values[0]
                    weeks_difference = (week - last_ranking_date).days // 7
                    imputed_rank = last_rank + weeks_difference * rank_penalty_per_week_inactivity
                    if imputed_rank > 3333: # Don't over-penalize
                        m.at[i, 'player_2_rank'] = 3333   # Set a default value   
                    else:
                        m.at[i, 'player_2_rank'] = imputed_rank
                        last_rankings[p2_id] = imputed_rank
                else: 
                    m.at[i, 'player_2_rank'] = 3333  # Set a default value
            else:
                m.at[i, 'player_2_rank'] = 3333  # Set a default value


# %% [markdown]
# Test the function using the example of Kevin Anderson:

# %%
# test function using only reduced dataset: Kevin Anderson
impute_missing_rankings(matches_processed_ka_df, rankings_processed_df)
matches_processed_ka_df[['tourney_date_dt', 'player_1_name', 'player_2_name', 'player_1_rank', 'player_2_rank','round', 'winner']]

# %%
rankings_processed_df[(rankings_processed_df['ranking_date'].between(20220501, 20230801)) & (rankings_processed_df['player'] == 104731)]

# %%
weeks_ka_inactive = (dt.date(2023,7,17) - dt.date(2022,5,23)).days  // 7 # no. or weeks inactivity
weeks_ka_inactive * 10 # rank place penalty of 10 per week

# %% [markdown]
# The penalty of roughly 600 places for a 60 week period of inactivity reflects roughly the output of the function impute_missing_rankings.
# 
# Finally, we apply the function to our full dataset, and do a small check to verify that no null values exist anymore for these features:

# %%
impute_missing_rankings(matches_processed_df, rankings_processed_df)

# %%
matches_processed_df['player_1_rank'].isna().value_counts()

# %%
matches_processed_df['player_2_rank'].isna().value_counts()

# %%
matches_processed_df[['player_1_rank', 'player_2_rank']].describe()

# %% [markdown]
# ## Feature Engineering

# %%
# make a new copy of the dataframe, for starting the feature engineering
matches_features_df = matches_processed_df.copy().reset_index()
matches_features_df.info()
# matches_features_df.to_csv("matches_features_df.csv", sep=',', header=True)

# %% [markdown]
# ### Add feature for cumulative number of games played so far in a tournament - OPTIONAL and WIP
# It would be interesting if the cumulative number of games played so far in a tournament could be used to predict the next result of a match, indicating either fatigue or dominance (won in straight sets). So far this feature is optional for our prediction model.
# 
# *Note: this function does not yet work 100%. It doesn't yet calculate the `player_x_tourney_cum_games_count` correctly in the case where a player can appear as player_1 or player_2 in the same tournament.*

# %%
# function calc_game_counts: to calculate game counts for each match, uses vectorization instead of row iteration for performance reasons
'''
def calc_game_counts(m):
    # Initialize columns to store game counts
    m['player_1_match_games_count'] = 0
    m['player_2_match_games_count'] = 0
    m['player_1_tourney_cum_games_count'] = 0
    m['player_2_tourney_cum_games_count'] = 0
    
    # Regular expression to match scores
    score_pattern = r'(\d+)-(\d+)(?:\(\d+\))?'
    
    # Extract individual scores using regular expression and convert to numeric values
    scores = m['score'].str.extractall(score_pattern).astype(int)
    m[['player_1_games', 'player_2_games']] = scores.groupby(level=0).sum()
    
    # Determine the winner and adjust game counts accordingly
    winner_mask = m['winner'] == 'player_1'
    m.loc[winner_mask, 'player_1_match_games_count'] = m.loc[winner_mask, 'player_1_games']
    m.loc[~winner_mask, 'player_1_match_games_count'] = m.loc[~winner_mask, 'player_2_games']
    
    m['player_2_match_games_count'] = m['player_1_games'] + m['player_2_games'] - m['player_1_match_games_count']
    
    # Calculate cumulative game counts using groupby and cumsum without resetting
    m['player_1_tourney_cum_games_count'] = m.groupby(['tourney_id', 'player_1_id'])['player_1_match_games_count'].cumsum() - m['player_1_match_games_count']
    m['player_2_tourney_cum_games_count'] = m.groupby(['tourney_id', 'player_2_id'])['player_2_match_games_count'].cumsum() - m['player_2_match_games_count']
    
    # Set the initial cumulative game counts to 0 for the first matches of each player
    m.loc[m.groupby(['tourney_id', 'player_1_id'])['player_1_match_games_count'].cumcount() == 0, 'player_1_tourney_cum_games_count'] = 0
    m.loc[m.groupby(['tourney_id', 'player_2_id'])['player_2_match_games_count'].cumcount() == 0, 'player_2_tourney_cum_games_count'] = 0
    
    return m
'''

# %%
# run the function calc_game_counts on the matches dataset
'''
matches_processed_df = calc_game_counts(matches_processed_df)
'''

# %%
# test by exporting to .csv
'''matches_processed_df[matches_processed_df['tourney_id'].str.contains('2000-451|2000-301')][['tourney_id','match_num', 'player_1_id','player_2_id',
													   'player_1_name','player_2_name', 'round' ,'score', 'winner', 
														'player_1_match_games_count', 'player_2_match_games_count',
														'player_1_tourney_cum_games_count', 'player_2_tourney_cum_games_count']].to_csv("matches_processed.csv", sep=',', header=True, index=False)
'''

# %%
# review the output of the new columns for the first 31 rows (1 tournament)
'''
matches_processed_df[['tourney_id','match_num', 'player_1_id','player_2_id',
													   'player_1_name','player_2_name', 'round' ,'score', 'winner', 
														'player_1_match_games_count', 'player_2_match_games_count',
														'player_1_tourney_cum_games_count', 'player_2_tourney_cum_games_count']].head(31)
'''

# %% [markdown]
# ### Add feature for ranking difference
# This feature may help our model more easily assess the how the ranking plays a factor in determining the winner of the match. It simply calculates the weight of the difference between player_2_rank and player_1_rank, by using a normalized difference. The normalized difference is expressed as a number between 0 and 1. In that case, the closer the ranking between player 1 and player 2, the higher the number will be.

# %%
# calculate max. possible rank difference
max_possible_rank_difference = max(matches_features_df['player_2_rank'] - matches_features_df['player_1_rank'])

# calculate normalized rank difference
matches_features_df['ranking_difference'] = 1 - ((matches_features_df['player_2_rank'] - matches_features_df['player_1_rank']) / max_possible_rank_difference)

# preview the result for the last 5 observations of the dataset
matches_features_df[['tourney_date_dt', 'player_1_name', 'player_1_rank','player_2_name', 'player_2_rank', 'ranking_difference']].tail(5)

# %%
# preview the result for the first 5 observations of the dataset
matches_features_df[['tourney_date_dt', 'player_1_name', 'player_1_rank','player_2_name', 'player_2_rank', 'ranking_difference']].head(5)

# %% [markdown]
# ### Add feature for cumulative matches played count, and win percentages per surface and tourney level for player 1 and player 2
# This cumulative matches played count and win percentages per surface and tourney level are important features for our prediction model.

# %% [markdown]
# #### Surface win %
# This feature will show a player's success so far on a particular tennis court surface. There will be a number expressed as a percentage which will reflect the number of wins divided by the total matches on a surface, prior to that match taking place.
# First, what are the different surfaces being played on?

# %%
# what are the different surfaces played on since 2000?
matches_features_df['surface'].value_counts(normalize=True)

# %%
# what are the different surfaces played on in the top 10 tournaments in 2000?
print(matches_features_df[matches_features_df['surface'] == 'Hard']['tourney_name'].value_counts().head(10))
print(matches_features_df[matches_features_df['surface'] == 'Clay']['tourney_name'].value_counts().head(10))
print(matches_features_df[matches_features_df['surface'] == 'Grass']['tourney_name'].value_counts().head(10))
print(matches_features_df[matches_features_df['surface'] == 'Carpet']['tourney_name'].value_counts().head(10))


# %%
# create a test dataset for all 4 surface types, and preview the columns and sample rows relevant for calculation
matches_4surfaces = matches_features_df[(matches_features_df['tourney_name'].isin(['Auckland', 'Barcelona', 'Halle', 'Basel']))
					& (matches_features_df['tourney_date'] < 20010000)][['tourney_name', 'tourney_date_dt', 'match_num', 'surface', 'tourney_level'
                                                                                , 'player_1_id', 'player_2_id', 'player_1_name', 'player_2_name'
                                                                                , 'winner'
					]]
matches_4surfaces

# %% [markdown]
# #### Tournament level win %
# This feature will show a player's success so far on a particular type (level) of tournament. There will be a number expressed as a percentage which will reflect the number of wins divided by the total matches on that level, prior to that match taking place.
# First, what are the different tournament level being played?  

# %%
# what are the different tournament levels played since 2000?
matches_features_df['tourney_level'].value_counts(normalize=True)

# %% [markdown]
# #### Function to calculate win %s

# %%
def calc_cum_match_counts_and_pct (df):

    # Initialize dictionaries to keep track of cumulative match counts and wins for each player and surface
    player_cumulative_counts = {}
    player_surface_cumulative_counts = {}
    player_surface_cumulative_wins = {}
    player_tourney_level_cumulative_counts = {}
    player_tourney_level_cumulative_wins = {}

    # Lists to store the cumulative match counts for each row
    player_1_cumulative_counts_list = []
    player_2_cumulative_counts_list = []
    player_1_surface_cumulative_counts_list = []
    player_2_surface_cumulative_counts_list = []
    player_1_surface_cumulative_wins_list = []
    player_2_surface_cumulative_wins_list = []
    player_1_tourney_level_cumulative_counts_list = []
    player_2_tourney_level_cumulative_counts_list = []
    player_1_tourney_level_cumulative_wins_list = []
    player_2_tourney_level_cumulative_wins_list = []

    for index, row in df.iterrows():
        player_1_id = row['player_1_id']
        player_2_id = row['player_2_id']
        surface = row['surface']
        tourney_level = row['tourney_level']

        # Get the cumulative match counts so far for each player
        player_1_cumulative_count = player_cumulative_counts.get(player_1_id, 0)
        player_2_cumulative_count = player_cumulative_counts.get(player_2_id, 0)

        # Get the cumulative match counts and wins on the current surface for each player
        player_1_surface_cumulative_count = player_surface_cumulative_counts.get((player_1_id, surface), 0)
        player_2_surface_cumulative_count = player_surface_cumulative_counts.get((player_2_id, surface), 0)
        player_1_surface_cumulative_wins = player_surface_cumulative_wins.get((player_1_id, surface), 0)
        player_2_surface_cumulative_wins = player_surface_cumulative_wins.get((player_2_id, surface), 0)

        # Get the cumulative match counts and wins on the current tourney level for each player
        player_1_tourney_level_cumulative_count = player_tourney_level_cumulative_counts.get((player_1_id, tourney_level), 0)
        player_2_tourney_level_cumulative_count = player_tourney_level_cumulative_counts.get((player_2_id, tourney_level), 0)
        player_1_tourney_level_cumulative_wins = player_tourney_level_cumulative_wins.get((player_1_id, tourney_level), 0)
        player_2_tourney_level_cumulative_wins = player_tourney_level_cumulative_wins.get((player_2_id, tourney_level), 0)

        # Update the cumulative match counts and wins for each player, surface and tourney level in the current players' lists
        player_1_cumulative_counts_list.append(player_1_cumulative_count)
        player_2_cumulative_counts_list.append(player_2_cumulative_count)
        player_1_surface_cumulative_counts_list.append(player_1_surface_cumulative_count)
        player_2_surface_cumulative_counts_list.append(player_2_surface_cumulative_count)
        player_1_tourney_level_cumulative_counts_list.append(player_1_tourney_level_cumulative_count)
        player_2_tourney_level_cumulative_counts_list.append(player_2_tourney_level_cumulative_count)

        # Calculate and update the cumulative match won percentage on the current surface for each player
        player_1_surface_cumulative_wins_percentage = (
            player_1_surface_cumulative_wins / player_1_surface_cumulative_count
        ) if player_1_surface_cumulative_count > 0 else 0.0
        player_2_surface_cumulative_wins_percentage = (
            player_2_surface_cumulative_wins / player_2_surface_cumulative_count
        ) if player_2_surface_cumulative_count > 0 else 0.0

        player_1_surface_cumulative_wins_list.append(player_1_surface_cumulative_wins_percentage)
        player_2_surface_cumulative_wins_list.append(player_2_surface_cumulative_wins_percentage)

        # Calculate and update the cumulative match won percentage on the current tourney level for each player
        player_1_tourney_level_cumulative_wins_percentage = (
            player_1_tourney_level_cumulative_wins / player_1_tourney_level_cumulative_count
        ) if player_1_tourney_level_cumulative_count > 0 else 0.0
        player_2_tourney_level_cumulative_wins_percentage = (
            player_2_tourney_level_cumulative_wins / player_2_tourney_level_cumulative_count
        ) if player_2_tourney_level_cumulative_count > 0 else 0.0

        player_1_tourney_level_cumulative_wins_list.append(player_1_tourney_level_cumulative_wins_percentage)
        player_2_tourney_level_cumulative_wins_list.append(player_2_tourney_level_cumulative_wins_percentage)

        # Increment the cumulative match counts and wins for each player and surface in the dictionaries
        player_cumulative_counts[player_1_id] = player_1_cumulative_count + 1
        player_cumulative_counts[player_2_id] = player_2_cumulative_count + 1
        player_surface_cumulative_counts[(player_1_id, surface)] = player_1_surface_cumulative_count + 1
        player_surface_cumulative_counts[(player_2_id, surface)] = player_2_surface_cumulative_count + 1
        player_tourney_level_cumulative_counts[(player_1_id, tourney_level)] = player_1_tourney_level_cumulative_count + 1
        player_tourney_level_cumulative_counts[(player_2_id, tourney_level)] = player_2_tourney_level_cumulative_count + 1

        # Increment the cumulative match wins on the current surface for the winner
        if row['winner'] == 'player_1':
            player_surface_cumulative_wins[(player_1_id, surface)] = player_1_surface_cumulative_wins + 1
        else:
            player_surface_cumulative_wins[(player_2_id, surface)] = player_2_surface_cumulative_wins + 1

        # Increment the cumulative match wins on the current tourney level for the winner
        if row['winner'] == 'player_1':
            player_tourney_level_cumulative_wins[(player_1_id, tourney_level)] = player_1_tourney_level_cumulative_wins + 1
        else:
            player_tourney_level_cumulative_wins[(player_2_id, tourney_level)] = player_2_tourney_level_cumulative_wins + 1

    # Add the cumulative match count and surface- and tourney level-related columns to the input dataset
    df['player_1_cum_match_count'] = player_1_cumulative_counts_list
    df['player_2_cum_match_count'] = player_2_cumulative_counts_list
    df['player_1_surface_cum_match_count'] = player_1_surface_cumulative_counts_list
    df['player_2_surface_cum_match_count'] = player_2_surface_cumulative_counts_list
    df['player_1_surface_cum_win_percentage'] = player_1_surface_cumulative_wins_list
    df['player_2_surface_cum_win_percentage'] = player_2_surface_cumulative_wins_list
    df['player_1_tourney_level_cum_match_count'] = player_1_tourney_level_cumulative_counts_list
    df['player_2_tourney_level_cum_match_count'] = player_2_tourney_level_cumulative_counts_list
    df['player_1_tourney_level_cum_win_percentage'] = player_1_tourney_level_cumulative_wins_list
    df['player_2_tourney_level_cum_win_percentage'] = player_2_tourney_level_cumulative_wins_list

    # Add win percentage difference columns for surface- and tourney level
    df['surface_win_pct_difference'] = df['player_1_surface_cum_win_percentage'] - df['player_2_surface_cum_win_percentage']
    df['tourney_level_win_pct_difference'] = df['player_1_tourney_level_cum_win_percentage'] - df['player_2_tourney_level_cum_win_percentage']
    
    return df


# %%
# Apply the function to the data
matches_features_df = calc_cum_match_counts_and_pct(matches_features_df)

# %% [markdown]
# #### Test the function on some data

# %%
# test surface win pct and tourney level win pct for one player
matches_features_df[(matches_features_df['player_1_name'] == 'Roger Federer') 
                     |
                     (matches_features_df['player_2_name'] == 'Roger Federer')][['tourney_name', 'tourney_date_dt', 'match_num', 'surface', 'tourney_level'
                                                                                , 'player_1_name', 'player_2_name'
                                                                                , 'winner'
                                                                                 , 'player_1_surface_cum_win_percentage','player_2_surface_cum_win_percentage'
                                                                                 , 'player_1_tourney_level_cum_win_percentage','player_2_tourney_level_cum_win_percentage'
                                                                                 , 'surface_win_pct_difference', 'tourney_level_win_pct_difference'
                                                                                 ]]

# %%
# test surface win pct and tourney level win pct for another player
matches_features_df[(matches_features_df['player_1_name'] == 'Thomas Enqvist') 
                    | (matches_features_df['player_2_name'] == 'Thomas Enqvist')][['tourney_name', 'tourney_date_dt', 'match_num', 'surface', 'tourney_level'
                                                                                , 'player_1_name', 'player_2_name'
                                                                                , 'winner'
                                                                                , 'player_1_surface_cum_win_percentage','player_2_surface_cum_win_percentage'
                                                                                , 'player_1_tourney_level_cum_win_percentage','player_2_tourney_level_cum_win_percentage'
                                                                                , 'surface_win_pct_difference', 'tourney_level_win_pct_difference']]

# %%
# test for 4 tournaments, each on different surface
matches_4surfaces_calc = calc_cum_match_counts_and_pct(matches_4surfaces)
matches_4surfaces_calc[['tourney_name', 'tourney_date_dt', 'match_num', 'surface'
                                                                                , 'player_1_name', 'player_2_name'
                                                                                , 'winner'
                                                                                 , 'player_1_surface_cum_win_percentage','player_2_surface_cum_win_percentage'
                                                                                 , 'surface_win_pct_difference', 'tourney_level_win_pct_difference']].head(10)

# %%
# create a test dataset for all 5 tournament levels, and preview the columns and sample rows relevant for calculation
matches_5levels = matches_features_df[(matches_features_df['tourney_name'].isin(['Auckland', 'Davis Cup QLS R1: GER vs SUI', 'Tour Finals', 'Australian Open', 'Indian Wells Masters',]))
					& (matches_features_df['tourney_date'] < 20010000)][['tourney_name', 'tourney_date_dt', 'match_num', 'tourney_level'
                                                                                , 'player_1_id', 'player_2_id', 'player_1_name', 'player_2_name'
                                                                                , 'winner'
                                                                                , 'player_1_tourney_level_cum_win_percentage','player_2_tourney_level_cum_win_percentage'
                                                                                , 'tourney_level_win_pct_difference'
					                                                    ]]
matches_5levels

# %% [markdown]
# ### Head-to-head
# 

# %% [markdown]
# This function calculates the head-to-head record for two players, and expresses the result as a percentage. 

# %%
def calc_h2h_win_pct(df):


    # Create a dictionary to store cumulative wins and matches for each pair of players
    h2h_stats = {}

    # Initialize new columns
    df['player_1_h2h_win_pct'] = 0.0
    df['player_2_h2h_win_pct'] = 0.0

    # Calculate head-to-head win percentage
    for index, row in df.iterrows():
        player_1_id = row['player_1_id']
        player_2_id = row['player_2_id']
        winner = row['winner']

        # Create a unique key for the pair of players
        player_pair_key = tuple(sorted([player_1_id, player_2_id]))

        # Update head-to-head stats for the player pair
        h2h_stats[player_pair_key] = h2h_stats.get(player_pair_key, {'ppk_1_wins': 0, 'ppk_2_wins': 0, 'matches': 0}) # ppk stands for "player pair key"

        # Calculate and update head-to-head win percentages
        if h2h_stats[player_pair_key]['matches'] == 0:
            # At the first match, both win percentages are set to 0
            df.at[index, 'player_1_h2h_win_pct'] = 0.0
            df.at[index, 'player_2_h2h_win_pct'] = 0.0
        else:
            # For subsequent matches, calculate based on the previous match
            if player_1_id == player_pair_key[0]: 
                player_1_win_pct = h2h_stats[player_pair_key]['ppk_1_wins'] / h2h_stats[player_pair_key]['matches']
            else: 
                player_1_win_pct = h2h_stats[player_pair_key]['ppk_2_wins'] / h2h_stats[player_pair_key]['matches']
            df.at[index, 'player_1_h2h_win_pct'] = player_1_win_pct
            df.at[index, 'player_2_h2h_win_pct'] = 1.0 - player_1_win_pct

        # Update head-to-head stats for the player pair after the match
        h2h_stats[player_pair_key]['matches'] += 1
        if ((winner == 'player_1') & (player_1_id == player_pair_key[0]) | (winner == 'player_2') & (player_2_id == player_pair_key[0])):
            h2h_stats[player_pair_key]['ppk_1_wins'] += 1
        else:
            h2h_stats[player_pair_key]['ppk_2_wins'] += 1

    return df

# %%
# Apply the function to the data
matches_features_df = calc_h2h_win_pct(matches_features_df)

# %%
# Calculate a h2h difference
matches_features_df['h2h_win_pct_difference'] = matches_features_df['player_1_h2h_win_pct'] - matches_features_df['player_2_h2h_win_pct']

# %%
# create a test dataset for 2 players' head-to-head matches, and preview the columns and sample rows relevant for calculation

matches_2players = matches_features_df[((matches_features_df['player_1_name'] == 'Andrey Rublev') & (matches_features_df['player_2_name'] == 'Jannik Sinner')) | 
                 ((matches_features_df['player_1_name'] == 'Jannik Sinner') & (matches_features_df['player_2_name'] == 'Andrey Rublev'))][['tourney_name', 'tourney_date_dt', 'match_num', 'tourney_level', 'round'
                                                                                                                                        , 'player_1_id', 'player_2_id'
                                                                                                                                        , 'player_1_name', 'player_2_name'
                                                                                                                                        , 'player_1_rank', 'player_2_rank'
                                                                                                                                        , 'winner'
                                                                                                                                        , 'player_1_h2h_win_pct','player_2_h2h_win_pct'
                                                                                                                                        , 'h2h_win_pct_difference'
                                                                                                                                        ]]
matches_2players

# %% [markdown]
# ### Recent form

# %% [markdown]
# ## Data Analysis

# %% [markdown]
# Some basics first before moving to the prediction models: what features are we left with after data processing and feature engineering?

# %%
matches_features_df.info()

# %% [markdown]
# We have engineered so far these features for assisting our prediction model:
# - ranking_difference
# - player_1_surface_cum_win_percentage     
# - player_2_surface_cum_win_percentage
# - surface_win_pct_difference
# - player_1_tourney_level_cum_win_percentage
# - player_2_tourney_level_cum_win_percentage
# - tourney_level_win_pct_difference
# - player_1_h2h_win_pct 
# - player_2_h2h_win_pct
# - h2h_win_pct_difference
# 
# The majority of the other features in our dataset are probably not needed,  # , 'player_1_tourney_level_cum_win_percentage'
#                                                         # , 'player_2_tourney_level_cum_win_percentage'and when applying encoding, we'll end up with a lot of additional useless features. 
# So as a final step, we remove unused features from out dataset.

# %%
matches_features_df.isnull().sum()

# %%
# remove data for the year 2000, so the majority of features with values = 0 is removed
matches_features_df = matches_features_df[matches_features_df['tourney_date_dt'].dt.year > 2000]
matches_features_df.groupby(matches_features_df['tourney_date_dt'].dt.year)['tourney_id'].count()

# %%
matches_features_trimmed_df = matches_features_df.drop(columns=[
                                                        'tourney_id'
                                                        , 'tourney_name'
                                                        , 'surface'
                                                        , 'draw_size'
                                                        , 'tourney_level'
                                                        , 'tourney_date'
                                                        , 'match_num'
                                                        , 'score'
                                                        , 'best_of'
                                                        , 'round'
                                                        , 'minutes'
                                                        , 'player_1_name'
                                                        , 'player_2_name'
                                                        , 'player_1_id'
                                                        , 'player_2_id'
                                                        , 'player_1_seed'
                                                        , 'player_2_seed'
                                                        , 'player_1_entry'
                                                        , 'player_2_entry'
                                                        , 'player_1_hand'
                                                        , 'player_2_hand'
                                                        , 'player_1_ht'
                                                        , 'player_2_ht'
                                                        , 'player_1_ioc'
                                                        , 'player_2_ioc'
                                                        , 'player_1_age'
                                                        , 'player_2_age'
                                                        , 'player_1_rank'
                                                        , 'player_2_rank'
                                                        , 'player_1_rank_points'
                                                        , 'player_2_rank_points'
                                                        # , 'winner'
                                                        , 'tourney_date_dt'
                                                        , 'tourney_date_dt_preceding_monday'
                                                        #, 'ranking_difference'
                                                        , 'player_1_cum_match_count'
                                                        , 'player_2_cum_match_count'
                                                        , 'player_1_surface_cum_match_count'
                                                        , 'player_2_surface_cum_match_count'
                                                        , 'player_1_surface_cum_win_percentage'
                                                        , 'player_2_surface_cum_win_percentage'
                                                        # , 'tourney_level_win_pct_difference'
                                                        , 'player_1_tourney_level_cum_match_count'
                                                        , 'player_2_tourney_level_cum_match_count'
                                                        , 'player_1_tourney_level_cum_win_percentage'
                                                        , 'player_2_tourney_level_cum_win_percentage'
                                                        #, 'tourney_level_win_pct_difference'
                                                        , 'player_1_h2h_win_pct'
                                                        , 'player_2_h2h_win_pct'
                                                        # , 'h2h_win_pct_difference'
                                                        ], axis=1)
matches_features_trimmed_df.info()

# %%
# make a new copy of the dataframe, prior to starting the prediction
matches_pred_df = matches_features_trimmed_df.copy().reset_index(drop=True)
matches_pred_df.info()

# %%
#encode categorical data
train = pd.get_dummies(matches_pred_df, drop_first=True)
train.head(200)

# %%
train.info()

# %%
# convert data type of label from bool to int64
train['winner_player_2'] = train['winner_player_2'].astype('int64')
train.info()

# %%
import seaborn as sns

# Determine correlations between variables in the dataset
corr = train.corr()

# Plot the correlations
ax = sns.heatmap(corr, annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation'})
ax.set_title("Correlations of variables in the full dataset")
plt.show()

# %% [markdown]
# ## Prediction

# %% [markdown]
# In this section, we apply several different prediction models to determine which one gives us the most accurate results for predicting the outcome of a match.

# %% [markdown]
# ### Define some helper functions

# %% [markdown]
# #### Actual price vs predicted plot

# %%
def actual_vs_predicted_plot(y_true, y_pred):
  min_value=np.array([y_true.min(), y_pred.min()]).min()
  max_value= min=np.array([y_true.max(), y_pred.max()]).max()
  fig = plt.figure()
  ax = fig.gca()
  ax.scatter(y_true,y_pred, color="blue")
  ax.plot([min_value,max_value], [min_value, max_value], lw=4, color="green")
  ax.set_xlabel('Actual')
  ax.set_ylabel('Predicted')
  plt.xlim=0
  plt.ylim=0
  plt.show()

# %% [markdown]
# #### ROC plot

# %%
def plot_ROC(model, X_test, y_test):
  from sklearn.metrics import RocCurveDisplay
  tree_ROC = RocCurveDisplay.from_estimator(model, X_test, y_test, color='green', linewidth=3)
  plt.title('ROC Curve')
  plt.xlabel('False Alarm (1 - Specificity)')
  plt.ylabel('Recall (Sensitivity)')
  plt.show()

# %% [markdown]
# #### Plot confusion matrix

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_test, y_pred):

    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Player_2_winner = 0', 'Player_2_winner = 1'],
                yticklabels=['Player_2_winner = 0', 'Player_2_winner = 1'])
    plt.xlabel('Predicted Outcome')
    plt.ylabel('Actual Outcome')
    plt.title('Confusion Matrix')
    plt.show()

# %% [markdown]
# #### Plot variable importances

# %%
def plot_variable_importance(model, X_train):
  importances = pd.Series(data=model.feature_importances_,
                          index=X_train.columns)
  importances.sort_values().plot(kind='barh', color="#00802F")
  plt.title('Features Importances')

# %% [markdown]
# ### Decision Tree Model

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Instantiate Model with cross validation
tree = DecisionTreeClassifier(criterion="entropy", max_depth=1, max_features = 8, min_samples_leaf = 1, min_samples_split = 2, random_state=1)

# Create Train Data
X = train.drop("winner_player_2", axis=1)
y = train["winner_player_2"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# fit model
tree.fit(X_train, y_train)

#make prediction
y_pred = tree.predict(X_test)

# get prediction probabilities
tree.predict_proba(X_test)

# Evaluate Model Performance - accuracy
acc = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc)

# %% [markdown]
# For a Decision Tree model, with 66.4% there is a slightly better accuracy than our benchmark of 65.6%.

# %%
plot_confusion_matrix(y_test, y_pred)

# %%
X.info()

# %%
y

# %%
plot_variable_importance(tree, X_train)

# %%
# use this when looking for the best combination of hyperparamers. 
# the below example serves the purpose of a cross validation

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# hyperparameters for GridSearchCV
parameters = {
            "max_features": [8, 16],
            'max_depth': range(1,2),
            "min_samples_split": [2, 3, 5], 
            'min_samples_leaf': [1, 10, 20]
            }

# make a scoring function for accuracy
acc_score = make_scorer(accuracy_score, greater_is_better=True)

# fit model
tree_model_CV = GridSearchCV(tree, parameters, scoring=acc_score, cv=5,verbose=3) # Apply 5 Cross Validiation Folds to find best hyperparameters
tree_model_CV.fit(X_train, y_train)

# %%
tree_model_CV.best_params_

# %% [markdown]
# ### Random Forest Model

# %%
from sklearn.ensemble import RandomForestClassifier

# Instantiate Model with cross validation
forest = RandomForestClassifier(max_depth=1, max_features = 8, min_samples_leaf = 1, min_samples_split = 2,  random_state=1)

# Create Train Data
X = train.drop("winner_player_2", axis=1)
y = train["winner_player_2"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# fit model
forest.fit(X_train, y_train)

#make prediction
y_pred = forest.predict(X_test)

# get prediction probabilities
forest.predict_proba(X_test)

# Evaluate Model Performance - accuracy
acc = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc)

# %%
plot_variable_importance(forest, X_train)

# %% [markdown]
# #### Grid Search - Random Forest
# 

# %%
# use this when looking for the best combination of hyperparamers. 
# the below example serves the purpose of a cross validation

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# hyperparameters for GridSearchCV
parameters = {
            "max_features": [8, 16],
            'max_depth': range(1,2),
            "min_samples_split": [2, 3, 5], 
            'min_samples_leaf': [1, 10, 20]
            }

# make a scoring function for accuracy
acc_score = make_scorer(accuracy_score, greater_is_better=True)

# fit model
forest_model_CV = GridSearchCV(forest, parameters, scoring=acc_score, cv=5,verbose=3) # Apply 5 Cross Validiation Folds to find best hyperparameters
forest_model_CV.fit(X_train, y_train)

# %%
forest_model_CV.best_params_

# %% [markdown]
# ### Gradient Booster Tree Model

# %%
from sklearn.ensemble import GradientBoostingClassifier

boostedtrees = GradientBoostingClassifier(max_depth= 2, max_features = 8, min_samples_leaf = 1, min_samples_split = 5, n_estimators = 1000, 
                                         # learning_rate= 0.5,
                                        random_state=1)

# Create Train Data
X = train.drop("winner_player_2", axis=1)
y = train["winner_player_2"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# fit model
boostedtrees.fit(X_train, y_train)

#make prediction
y_pred = boostedtrees.predict(X_test)

# get prediction probabilities
print(boostedtrees.predict_proba(X_test))

accuracy_score(y_test, y_pred)

###########
# Run on 11.12
# Accuracy score: 0.6736 
# (max_depth= 2, max_features = 8, min_samples_leaf = 1, min_samples_split = 5, n_estimators = 1000, random_state=1)#
###########

# %%
plot_ROC(boostedtrees, X_test, y_test)

# %%
plot_variable_importance(boostedtrees, X_train)

# %% [markdown]
# #### Grid Search - Gradient Boosted Trees
# 

# %%
# use this when looking for the best combination of hyperparamers. 
# the below example serves the purpose of a cross validation

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# hyperparameters for GridSearchCV
parameters = {
            'max_depth': [2],
            "max_features": [8],
            'min_samples_leaf': [1, 5, 10],
            "min_samples_split": [5], 
            'n_estimators': [1000, 2000]
            }

# make a scoring function for accuracy
acc_score = make_scorer(accuracy_score, greater_is_better=True)

# fit model
boosted_model_CV = GridSearchCV(boostedtrees, parameters, scoring=acc_score, cv=5,verbose=3) # Apply 5 Cross Validiation Folds to find best hyperparameters
boosted_model_CV.fit(X_train, y_train)

###########
# CV Run on 07.12 09:30
# Runtime 35 mins
# 
# {'max_depth': 2,
#  'max_features': 8,
#  'min_samples_leaf': 10,
#  'min_samples_split': 5,
#  'n_estimators': 1000}
# 
# ######### 

# %%
boosted_model_CV.best_params_

# %% [markdown]
# ## Final Conclusion  <a name="final-concl"></a>
# 
# ...


