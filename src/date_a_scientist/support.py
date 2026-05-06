import gc
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.sparse import vstack,csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None
    cpsp = None
HAS_CUDA_GPU = HAS_CUPY and cp.is_available()


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
    #print(f'\n{astro_sign.upper()} distribution:') # uncomment (1 of 2) if viewing the full distribution is desired
    for sign in df.sign.value_counts().index:
        if astro_sign in sign:
            #print(f'{sign}: {df.sign.value_counts()[sign]}') # uncomment (2 of 2) if viewing the full distribution is desired
            if 'matters a lot' in sign:
                matters_count = df.sign.value_counts()[sign]
            sign_count += df.sign.value_counts()[sign]
    print(f'  {astro_sign.title()}: {matters_count/sign_count * 100:.2f}%')
    return matters_count/sign_count, sign_count


def do_signs_matter(df):
    # INPUT: df = pandas dataframe
    # OUTPUT: calls sign_importance_distribution() for each astrological sign, prints the percentage of users who did not provide a sign.

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

    # New column for last_online as a datetime object
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
        #print(f'{state.title()}: {state_total}') # Uncomment to see the full state-by-state breakdown
        overall_total += state_total
        if state == 'california':
            california_total = state_total
    print('\nTotal USA based users:     ',overall_total)
    print('California based users:    ',california_total)
    print('Non-California USA users:  ',overall_total - california_total)
    print('International based users: ',df.shape[0] - overall_total, '\n')
    print('Percentage of users outside California:',f'{(df.shape[0] - california_total) / df.shape[0] * 100:.2f}%')
    return None


def keep_top_n_per_row(matrix, top_n=100):
    """
    INPUT: matrix = scipy sparse matrix, top_n = number of top values to keep per row
    OUTPUT: scipy sparse matrix with only the top_n values per row.

    Significantly reduces sparce matrix size and density.
    """
    if top_n is None:
        return matrix

    matrix = matrix.tocsr()

    rows = []
    cols = []
    data = []

    for row_idx in range(matrix.shape[0]):
        start = matrix.indptr[row_idx]
        end = matrix.indptr[row_idx + 1]

        row_cols = matrix.indices[start:end]
        row_data = matrix.data[start:end]

        if row_data.size == 0:
            continue

        keep = min(top_n, row_data.size)
        top_idx = np.argpartition(row_data, -keep)[-keep:]
        top_idx = top_idx[np.argsort(row_data[top_idx])[::-1]]

        rows.extend([row_idx] * keep)
        cols.extend(row_cols[top_idx])
        data.extend(row_data[top_idx])

    return csr_matrix(
        (data, (rows, cols)),
        shape=matrix.shape,
        dtype=np.float32
    )




def get_gpu_csr(df1, df2, stop_words = 'english', top_n=100):
    """ GPU BASED DOT PRODUCT
    INPUT: df1 = pandas dataframe, df2 = pandas dataframe, stop_words = list of stop words, top_n = number of top values to keep per row
    OUTPUT: dot product of df1 and df2 sparse matrices with only the top_n values per row.
    """
    if not HAS_CUDA_GPU:
        return print("No CUDA GPU detected.")

    print("CuPy version:", cp.__version__)


    # Fit the vectorizer on ALL comments to create the shared vocabulary.
    # This part still runs on CPU because TfidfVectorizer is scikit-learn-based.
    matchmaking_essays = pd.concat(
        [
            df1.combined_essays,
            df2.combined_essays
        ],
        ignore_index=True
    )

    vectorizer = TfidfVectorizer(stop_words=stop_words,dtype=np.float32)
    vectorizer.fit(matchmaking_essays)

    # Transform each group separately
    men_tfidf = vectorizer.transform(
        df1.combined_essays
    ).astype("float32")

    women_tfidf = vectorizer.transform(
        df2.combined_essays
    ).astype("float32")

    chunk_list = []
    women_tfidf_gpu = cpsp.csr_matrix(women_tfidf)

    for start in range(0, men_tfidf.shape[0], 1000):
        end = min(start + 1000, men_tfidf.shape[0])
        #print(f"Processing rows {start} to {end}")
        men_tfidf_chunk = cpsp.csr_matrix(men_tfidf[start:end])

        # Dot product between men(chunk) and women(whole)
        men_women_interaction_chunk = men_tfidf_chunk @ women_tfidf_gpu.T
        chunk_list.append(keep_top_n_per_row(men_women_interaction_chunk.get().tocsr(), top_n=top_n).tocsr())

        if men_women_interaction_chunk.nnz == 0:
            raise RuntimeError(
                f"GPU multiplication returned an empty result for rows {start} to {end}."
            )

        # Clean up intermediate objects
        del men_tfidf_chunk, men_women_interaction_chunk
        cp.get_default_memory_pool().free_all_blocks()


    men_women_csr = vstack(chunk_list,format="csr")
    print("\nmen_women_csr nnz:", men_women_csr.nnz)
    print(f'csr density: {men_women_csr.nnz / (men_women_csr.shape[0] * men_women_csr.shape[1])*100:.4f}%')

    del women_tfidf_gpu
    del men_tfidf
    del women_tfidf

    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    return men_women_csr



def get_cpu_csr(df1,df2,stop_words='english',top_n=100):
    """ CPU BASED DOT PRODUCT
    INPUT: df1 = pandas dataframe, df2 = pandas dataframe, stop_words = list of stop words, top_n = number of top values to keep per row
    OUTPUT: dot product of df1 and df2 sparse matrices with only the top_n values per row.
    """
    # Fit the vectorizer on ALL comments to create shared vocabulary
    matchmaking_essays = pd.concat(
        [
            df1.combined_essays,
            df2.combined_essays
        ],
        ignore_index=True
    )

    vectorizer = TfidfVectorizer(stop_words=stop_words,dtype=np.float32)
    vectorizer.fit(matchmaking_essays)

    # Transform each group separately
    men_tfidf = vectorizer.transform(
        df1.combined_essays
    ).astype('float32')   # Shape: (num_men, num_features)

    women_tfidf = vectorizer.transform(
        df2.combined_essays
    ).astype('float32')

    # Multiply Men (Users) by Women (Items) to get similarity scores
    # Resulting shape: (num_men, num_women)
    men_women_interaction = men_tfidf.dot(women_tfidf.T)

    # Convert to CSR format (required by implicit)
    men_women_csr = keep_top_n_per_row(men_women_interaction.tocsr(),top_n=top_n).tocsr()

    print("\nmen_women_csr nnz:", men_women_csr.nnz)
    print(f'csr density: {men_women_csr.nnz / (men_women_csr.shape[0] * men_women_csr.shape[1])*100:.4f}%')

    del men_tfidf,women_tfidf
    gc.collect()

    return men_women_csr



def measure_matrix_overlap(a,b):
    """
    INPUT: a = scipy sparse matrix, b = scipy sparse matrix
    OUTPUT: measures overlap between two sparse matrices. (1.0 = perfect overlap)
    """
    if isinstance(a, (list,np.ndarray)):
        a = csr_matrix(a)
    if isinstance(b, (list,np.ndarray)):
        b = csr_matrix(b)

    a = a.tocsr()
    b = b.tocsr()

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    overlaps = []

    for row_idx in range(a.shape[0]):
        a_start = a.indptr[row_idx]
        a_end = a.indptr[row_idx + 1]
        b_start = b.indptr[row_idx]
        b_end = b.indptr[row_idx + 1]

        a_cols = set(a.indices[a_start:a_end])
        b_cols = set(b.indices[b_start:b_end])

        if not a_cols and not b_cols:
            overlaps.append(1.0)
            continue

        union_size = len(a_cols | b_cols)

        if union_size == 0:
            overlaps.append(1.0)
        else:
            overlaps.append(len(a_cols & b_cols) / union_size)

    overlaps = np.array(overlaps)

    print("Mean row overlap:", overlaps.mean().round(6))
    print("Median row overlap:", np.median(overlaps).round(6))
    print("Min row overlap:", overlaps.min().round(6))
    print("Max row overlap:", overlaps.max().round(6))
    print("% of Rows with perfect overlap:", np.mean(overlaps == 1.0).round(6) * 100, "%")

    return overlaps



def measure_value_overlap(a: np.ndarray, b: np.ndarray):
    """
    INPUT: a = ndarray, b = ndarray
    OUTPUT: None, prints measure of overlap between ndarrays
    """

    # Make sure dimensions match
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    row_exact_scores = []
    row_shared_values_scores = []

    for i in range(a.shape[0]):
        # Retrieve row idx as a dense array
        a_row = a[i]
        b_row = b[i]

        # Distinct value overlap
        a_row_set = set(a_row)
        b_row_set = set(b_row)

        row_shared_values = len(a_row_set.intersection(b_row_set))
        row_shared_score = row_shared_values/len(a_row) if len(a_row) > 0 else 1.0
        row_shared_values_scores.append(row_shared_score)

        # Count exact matches for this row
        exact_matches = np.sum(a_row == b_row)
        exact_score = exact_matches / a_row.size if a_row.size > 0 else 1.0
        row_exact_scores.append(exact_score)

    a_set = set(a.ravel())
    b_set = set(b.ravel())
    shared_values = len(a_set.intersection(b_set))

    print(f"Mean percent of exact overlap per row: {(np.mean(row_exact_scores) * 100):.4f}%")
    print(f'Mean percent of values shared per row: {(np.mean(row_shared_values_scores) * 100):.4f}%')
    print(f"Total number of values shared: {shared_values} of {np.max([pd.Series(a.ravel()).nunique(),pd.Series(b.ravel()).nunique()])}")

    return None


