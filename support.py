import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def unique_and_missing_values(df):
    # INPUT: df = pandas dataframe
    # OUTPUT: pandas dataframe with unique values and missing values by column

    col_list = []
    print('\nNumber of unique and missing values by column:\n')
    for column in df.columns:
        if "essay" in column: # Exclude essays from unique value count
            continue
        col_list.append(column)
    unique_val = df[col_list].nunique()
    missing_val = df[col_list].isna().sum()
    combined_val = pd.DataFrame(zip(col_list,unique_val,missing_val),columns=np.array(['column','unique_values','missing_values'])).set_index('column')
    return combined_val


def sign_importance_distribution(df,astro_sign):
    # INPUT: df = pandas dataframe, astro_sign = string of astrological sign
    # OUTPUT: percentage of users who said the astrological sign matters, number of users with that sign.

    sign_count = 0
    matters_count = 0
    #print(f'\n{astro_sign.upper()} distribution:') # uncomment (1 of 2) if viewing full distribution is desired
    for sign in df.sign.value_counts().index:
        if astro_sign in sign:
            #print(f'{sign}: {df.sign.value_counts()[sign]}') # uncomment (2 of 2) if viewing full distribution is desired
            if 'matters a lot' in sign:
                matters_count = df.sign.value_counts()[sign]
            sign_count += df.sign.value_counts()[sign]
    print(f'  {astro_sign.title()}: {matters_count/sign_count * 100:.2f}%')
    return matters_count/sign_count, sign_count


def do_signs_matter(df):
    # INPUT: df = pandas dataframe
    # OUTPUT: calls sign_importance_distribution() for each astrological sign, prints percentage of users who did not provide a sign.

    print('Astrological Sign \'matters a lot\':')
    for sign in ['aries','aquarius','cancer','capricorn','gemini','leo','libra','pisces','sagittarius','scorpio','taurus','virgo']:
        sign_importance_distribution(df,astro_sign = sign)
    print(f'\n% Users who did not provide a sign: {df.sign.isna().sum() / df.shape[0] * 100:.2f}%')


def ohe_religion(df):
    # INPUT: df = pandas dataframe
    # OUTPUT: pandas dataframe with one-hot-encoded religious columns

    # Fill missing values with 'none'
    df.religion = df.religion.fillna('none')

    # ADD MANUAL OHE of User religious importance
    df['religion_serious'] = df.religion.apply(lambda x: True if 'very serious' in x else False)
    df['religion_somewhat'] = df.religion.apply(lambda x: True if 'somewhat serious' in x else False)
    df['religion_little'] = df.religion.apply(lambda x: True if 'not too serious' in x else False)
    df['religion_laughing'] = df.religion.apply(lambda x: True if 'laughing' in x else False)
    df['religion_none'] = df.religion.apply(lambda x: True if 'none' in x else False)

    # ADD Column for User religious affiliation
    # The first word in the 'religion' column is the religious affiliation
    df['religion_affiliation'] = df.religion.apply(lambda x: x.split()[0] if isinstance(x,str) else 'none')
    return df


def print_last_online(df):
    # INPUT: df = pandas dataframe
    # OUTPUT: prints number of users last_online by year, prints number of users last_online today.

    last_online_2011 = 0
    last_online_2012 = 0
    last_online_other = 0
    last_online_today = 0

    time_now = datetime.strptime(df.last_online.max(),'%Y-%m-%d-%H-%M') # setting time_now to last_online max.
    time_now_str = time_now.strftime('%Y-%m-%d')

    for date_time_str in df.last_online:
        if "2011" in date_time_str:
            last_online_2011 += 1
            continue
        if "2012" in date_time_str:
            last_online_2012 += 1
            if time_now_str in date_time_str:
                last_online_today += 1
                continue
            continue
        else:
            last_online_other += 1

    print(f'TODAY IS: {time_now_str}\n')

    print('Users last_online by year:')
    print("2012:",last_online_2012)
    if last_online_2011:
        print("2011:",last_online_2011)
    if last_online_other:
        print("Other:",last_online_other)

    print("\nToday\'s user count:",last_online_today)
    return None


def last_online_priority(df, priority_function):
    # INPUT: df = pandas dataframe
    # OUTPUT: pandas dataframe with new column for last_online_date, last_online_weeks, and last_online_priority

    # "Current time"
    time_now = datetime.strptime(df.last_online.max(),'%Y-%m-%d-%H-%M')

    # New column for last_online as datetime object
    df['last_online_date'] = df.last_online.apply(lambda x: datetime.strptime(x,'%Y-%m-%d-%H-%M'))

    # Number of weeks since last online
    df['last_online_weeks'] = (time_now - df.last_online_date).apply(lambda x: round(x.days/7))

    # Priority function using last_online_weeks (prioritizes users who have used the app recently)
    df['last_online_priority'] = df.last_online_weeks.apply(priority_function)
    return df

def plot_last_online_priority(df, priority_function, uindex):
    # INPUT: df = pandas dataframe, priority_function = function, uindex = index of user to plot
    # OUTPUT: Plot last_online_weeks vs. last_online_priority for user at index uindex

    # Plot helps visualize the last_online_priority value assignment:
    x_plot = pd.Series(range(29))
    y_plot = x_plot.apply(priority_function)
    plt.figure(figsize=(6,3))
    plt.title('Priority Weighting for Last Online',color='lightblue',fontsize=16)
    plt.xlabel('Weeks since last online')
    plt.ylabel('Priority value')
    plt.plot(x_plot,y_plot,color='lightcoral',linestyle='--',linewidth=2)
    plt.plot(df.last_online_weeks.iloc[uindex],df.last_online_priority.iloc[uindex],marker='o',color='lightgreen',markersize=10)
    plt.show()
    plt.close('all')

    print(f'Example for User @ index {uindex}')
    print(f'Weeks since last online:',df.last_online_weeks.iloc[uindex],'\n')
    print(f'Priority value: ',df.last_online_priority.iloc[uindex],'\n')
    return None


usa_states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
              "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
              "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
              "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
              "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
              "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]

def user_location_support(df):
    """
    INPUT: df = pandas dataframe
    OUTPUT: prints total number of users, users in California, users outside California, and international users.

    Most states have very low representation and will not support finding quality matches.
    Note some specific minor issues exist with this method:
    1. Cities with "state" names (i.e., Nevada City, California) will count towards Nevada State as well as California State.
    2. States with cardinal directions may count for both (i.e., Virginia user count will also include West Virginia users).
    """
    california_total = 0
    overall_total = 0
    for state in usa_states:
        state = state.lower()
        state_total = 0
        for city_state_region in df.location.unique():
            if state in city_state_region:
                state_total += df.location.value_counts().loc[city_state_region]
        #print(f'{state.title()}: {state_total}') # Uncomment to see full state-by-state breakdown
        overall_total += state_total
        if state == 'california':
            california_total = state_total
    print('\nTotal USA based users:     ',overall_total)
    print('California based users:    ',california_total)
    print('Non-California USA users:  ',overall_total - california_total)
    print('International based users: ',df.shape[0] - overall_total, '\n')
    print('Percentage of users outside California:',f'{(df.shape[0] - california_total) / df.shape[0] * 100:.2f}%')
    return None


