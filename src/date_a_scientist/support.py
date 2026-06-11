import gc
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.sparse import vstack,csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


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

    # CHECK if hardware supports CUDA
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
    df1_tfidf = vectorizer.transform(
        df1.combined_essays
    ).astype("float32")

    df2_tfidf = vectorizer.transform(
        df2.combined_essays
    ).astype("float32")

    chunk_list = []
    df2_tfidf_gpu = cpsp.csr_matrix(df2_tfidf)

    # USING iterations of data "chunks" to prevent memory issues. 1000 rows at a time.
    for start in range(0, df1_tfidf.shape[0], 1000):
        end = min(start + 1000, df1_tfidf.shape[0])
        #print(f"Processing rows {start} to {end}")
        df1_tfidf_chunk = cpsp.csr_matrix(df1_tfidf[start:end])

        # Dot product between df1(chunk) and df2(whole)
        df1_df2_interaction_chunk = df1_tfidf_chunk @ df2_tfidf_gpu.T
        chunk_list.append(keep_top_n_per_row(df1_df2_interaction_chunk.get().tocsr(), top_n=top_n).tocsr())

        if df1_df2_interaction_chunk.nnz == 0:
            raise RuntimeError(
                f"GPU multiplication returned an empty result for rows {start} to {end}."
            )

        # Clean up intermediate objects
        del df1_tfidf_chunk, df1_df2_interaction_chunk
        cp.get_default_memory_pool().free_all_blocks()


    df1_df2_csr = vstack(chunk_list,format="csr")
    print("\ndf1_df2_csr nnz:", df1_df2_csr.nnz)
    print(f'csr density: {df1_df2_csr.nnz / (df1_df2_csr.shape[0] * df1_df2_csr.shape[1])*100:.4f}%')

    del df2_tfidf_gpu
    del df1_tfidf
    del df2_tfidf

    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    return df1_df2_csr



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
    df1_tfidf = vectorizer.transform(
        df1.combined_essays
    ).astype('float32')   # Shape: (num_df1, num_features)

    df2_tfidf = vectorizer.transform(
        df2.combined_essays
    ).astype('float32')

    # Multiply df1 (Users) by df2 (Items) to get similarity scores
    # Resulting shape: (num_df1, num_df2)
    df1_df2_interaction = df1_tfidf.dot(df2_tfidf.T)

    """NEW"""
    if df1.shape == df2.shape: # If comparing to self, remove diagonal
        df1_df2_interaction.setdiag(0)
        df1_df2_interaction.eliminate_zeros()
    """NEW"""

    # Convert to CSR format (required by implicit)
    df1_df2_csr = keep_top_n_per_row(df1_df2_interaction.tocsr(),top_n=top_n).tocsr()

    print("\ndf1_df2_csr nnz:", df1_df2_csr.nnz)
    print(f'csr density: {df1_df2_csr.nnz / (df1_df2_csr.shape[0] * df1_df2_csr.shape[1])*100:.4f}%')

    del df1_tfidf,df2_tfidf
    gc.collect()

    return df1_df2_csr



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


def combine_ranked_matches(
        essay_matches,
        profile_matches,
        essay_weight = 0.6,
        profile_weight = 0.4,
        final_n = 50
):
    """
    Combine ranked matches from essay and profile matches using specified weights and return the top final_n candidates.
    INPUT: essay_matches = list(),
           profile_matches = list(),
           essay_weight = float,
           profile_weight = float,
           final_n = int
    OUTPUT: ranked_candidates = list()
    """
    profile_scores = {}

    for rank, candidate_idx in enumerate(essay_matches):
        rank_score = essay_weight * (len(essay_matches) - rank)
        profile_scores[candidate_idx] = profile_scores.get(candidate_idx, 0) + rank_score

    for rank, candidate_idx in enumerate(profile_matches):
        rank_score = profile_weight * (len(profile_matches) - rank)
        profile_scores[candidate_idx] = profile_scores.get(candidate_idx, 0) + rank_score

    ranked_candidates = sorted(
        profile_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [candidate_idx for candidate_idx, score in ranked_candidates[:final_n]]


def get_user_ids(recommendation_indices, df_in):
    """
    Return a list of user IDs corresponding to the given indices from the input DataFrame.
    INPUT: recommendation_indices = list(),
           df_in = pandas.DataFrame()
    OUTPUT: user_ids = list()
    """
    return list(df_in.user_id.values[recommendation_indices])


def show_recommended_profiles(source_df, target_df, source_idx, n=5):
    """
    Display recommended profiles based on the given source and target DataFrames.
    INPUT: source_df = pandas.DataFrame(),
           target_df = pandas.DataFrame(),
           source_idx = int,
           n = int (Number of recommended profiles to display)
    OUTPUT: slice from target_df corresponding to the recommended profiles.
    """
    recommended_indices = source_df.loc[source_idx, 'recommended_matches'][:n]
    display_columns = [
        'user_id',
        'age',
        'body_type',
        'height',
        'smokes_yes',
        'smokes_no',
        'religion_affiliation',
        'last_online_priority',
        #'education',
    ]
    print(f'User ID: {[int(val) for val in get_user_ids([source_idx], source_df)]} \t (user df index: {source_idx})')
    print(f'Recommended User IDs: {[int(val) for val in get_user_ids(recommended_indices, target_df)]}\n')
    return target_df.loc[recommended_indices, display_columns]



# Weighting features like 'age' to prioritize matching users close in age.
class WeightedNumericFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer that applies weighted scaling to numeric features.
    """
    def __init__(self, feature_weights=None):
        self.scaler_ = None
        self.columns_ = None
        self.feature_weights = feature_weights or {}

    def fit(self, x, y=None):
        self.columns_ = list(x.columns)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(x)
        return self

    def transform(self, x):
        x_scaled = self.scaler_.transform(x)

        for feature_name, weight in self.feature_weights.items():
            if feature_name in self.columns_:
                feature_index = self.columns_.index(feature_name)
                x_scaled[:, feature_index] *= weight

        return x_scaled


# noinspection DuplicatedCode
def match_same_sex(df_in,i=11):
    """
    Return DataFrame with new features corresponding to Profile, Essay, and Combined Recommendations.
    INPUT: df_in = pandas.DataFrame(), i = int (index of user to compare)
    OUTPUT: pandas.DataFrame() with new features added.
    """
    df_1 = df_in

    # ESSAY NEAREST NEIGHBORS:
    essay_nn = NearestNeighbors(
        n_neighbors=101,
        metric='cosine',
        algorithm='brute',
        n_jobs=-1
    )

    vectorizer = TfidfVectorizer(
        stop_words='english',
    )

    tfidf_df_1 = vectorizer.fit_transform(df_1.combined_essays)
    essay_nn.fit(tfidf_df_1)
    dist, idx = essay_nn.kneighbors(tfidf_df_1)

    essay_indices = idx[:,1:101]
    essay_scores = 1 - dist[:,1:101]

    # ADD essay match indices as new feature:
    df_1['essay_matches'] = list(essay_indices)

    # DROP features prior to profile (non-essay) matching
    list_of_features_to_drop = ['essay0', 'essay1','essay2', 'essay3', 'essay4', 'essay5', 'essay6', 'essay7', 'essay8',
                                'essay9','ethnicity', 'orientation', 'pets', 'sex', 'diet', 'last_online_date',
                                'last_online_weeks','ready_to_match','profile_matches','recommended_matches',
                                'recommended_matches_user_ids']

    for feature in list_of_features_to_drop:
        try:
            df_1.drop(columns=[feature],inplace=True)
        except KeyError:
            pass

    # ADD `df_profile_mm` for Profile Matchmaking and DROP user_id and essay features
    features_to_drop = ['user_id','essay_matches','combined_essays']
    df_profile_mm = df_1.drop(columns=features_to_drop)

    num_features = df_profile_mm.select_dtypes(include=['int64','int32','float64','float32']).columns.tolist()
    cat_features = df_profile_mm.select_dtypes(include=['object','str','bool','category']).columns.tolist()

    # profile_preprocessor uses WeightedNumericFeatures to prioritize matching users closer in age.
    profile_preprocessor = ColumnTransformer(
        transformers=[
            ('num',
             WeightedNumericFeatures(feature_weights={
                 'age': 3.0, # Higher weight assigned for the 'age' feature to prioritize matching users closer in age.
                 'height': 1.0,
                 'last_online_priority': 1.0,
                 'religion_affiliation': 1.0
             }),
             num_features),
            ('cat', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output = True
            ),
             cat_features)
        ],
        remainder='drop'
    )
    profile_matrix = profile_preprocessor.fit_transform(df_profile_mm)

    # PROFILE NEAREST NEIGHBORS:
    profile_nn = NearestNeighbors(
        n_neighbors=101,
        metric='cosine',
        algorithm='brute',
        n_jobs=-1
    )
    profile_nn.fit(profile_matrix)

    dist, idx = profile_nn.kneighbors(profile_matrix, return_distance=True)

    # ADD feature for profile matches
    df_1['profile_matches'] = list(idx[:,1:101])

    # ADD feature for combined recommendation
    df_1['recommended_matches'] = df_1.apply(
        lambda row: combine_ranked_matches(
            essay_matches = row.essay_matches,
            profile_matches = row.profile_matches,
            essay_weight = 0.2,
            profile_weight = 0.8,
            final_n = 50
        ),
        axis=1
    )

    # ADD feature for combined recommendation user ids
    df_1['recommended_matches_user_ids'] = df_1.apply(
        lambda row: get_user_ids(
            recommendation_indices = row.recommended_matches,
            df_in = df_1
        ),
        axis=1
    )

    # Compare User at index (i) with their top 5 matches
    print('User Profile:')
    print(df_1.loc[i,(
        'user_id',
        'age',
        'body_type',
        'height',
        'smokes_yes',
        'smokes_no',
        'religion_affiliation',
        'last_online_priority'
    )].T)

    print(show_recommended_profiles(df_1, df_1, i, n=5).T)

    return df_1


# noinspection DuplicatedCode
def match_opposite_sex(df_1,df_2,i=11):
    """
    Return DataFrame with new features corresponding to Profile, Essay, and Combined Recommendations.
    INPUT: df_1 = pandas.DataFrame(), df_2 = pandas.DataFrame(), i = int (index of user to compare)
    OUTPUT: df_1 and df_2 with new features added.
    """
    # ESSAY NEAREST NEIGHBORS:
    essay_nn = NearestNeighbors(
        n_neighbors=100,
        metric='cosine',
        algorithm='brute',
        n_jobs=-1
    )
    # ESSAY VECTORIZER:
    vectorizer = TfidfVectorizer(
        stop_words='english',
    )

    vectorizer.fit(pd.concat([df_1.combined_essays,df_2.combined_essays]))
    tfidf_df_1 = vectorizer.transform(df_1.combined_essays)
    tfidf_df_2 = vectorizer.transform(df_2.combined_essays)

    # FIT on target dataframe, QUERY from source dataframe
    essay_nn.fit(tfidf_df_2)  # Fit on df_2 (women)
    dist1, idx1 = essay_nn.kneighbors(tfidf_df_1)  # Query from df_1 (men)

    essay_nn.fit(tfidf_df_1)  # Fit on df_1 (men)
    dist2, idx2 = essay_nn.kneighbors(tfidf_df_2)  # Query from df_2 (women)

    essay_indices1 = idx1
    essay_scores1 = 1 - dist1
    essay_indices2 = idx2
    essay_scores2 = 1 - dist2

    # ADD essay match indices as new feature:
    df_1['essay_matches'] = list(essay_indices1)
    df_2['essay_matches'] = list(essay_indices2)

    # DROP features prior to profile (non-essay) matching
    list_of_features_to_drop = ['essay0', 'essay1','essay2', 'essay3', 'essay4', 'essay5', 'essay6', 'essay7', 'essay8',
                                'essay9','ethnicity', 'orientation', 'pets', 'sex', 'diet', 'last_online_date',
                                'last_online_weeks','ready_to_match','profile_matches','recommended_matches',
                                'recommended_matches_user_ids']

    for feature in list_of_features_to_drop:
        try:
            df_1.drop(columns=[feature],inplace=True)
            df_2.drop(columns=[feature],inplace=True)
        except KeyError:
            pass

    # ADD `df_profile_mm` for Profile Matchmaking and DROP user_id and essay features
    features_to_drop = ['user_id','essay_matches','combined_essays']
    df_profile_1 = df_1.drop(columns=features_to_drop)
    df_profile_2 = df_2.drop(columns=features_to_drop)

    num_features = df_profile_1.select_dtypes(include=['int64','int32','float64','float32']).columns.tolist()
    cat_features = df_profile_1.select_dtypes(include=['object','str','bool','category']).columns.tolist()

    # profile_preprocessor uses WeightedNumericFeatures to prioritize matching users closer in age.
    profile_preprocessor = ColumnTransformer(
        transformers=[
            ('num',
             WeightedNumericFeatures(feature_weights={
                 'age': 3.0, # Higher weight assigned for the 'age' feature to prioritize matching users closer in age.
                 'height': 1.0,
                 'last_online_priority': 1.0,
                 'religion_affiliation': 1.0
             }),
             num_features),
            ('cat', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output = True
            ),
             cat_features)
        ],
        remainder='drop'
    )
    profile_preprocessor.fit(pd.concat([df_profile_1,df_profile_2],axis=0,ignore_index=True))
    profile_matrix1 = profile_preprocessor.transform(df_profile_1)
    profile_matrix2 = profile_preprocessor.transform(df_profile_2)

    del df_profile_1,df_profile_2

    # PROFILE NEAREST NEIGHBORS:
    profile_nn = NearestNeighbors(
        n_neighbors=100,
        metric='cosine',
        algorithm='brute',
        n_jobs=-1
    )
    profile_nn.fit(profile_matrix2)  # Fit on women
    dist1, idx1 = profile_nn.kneighbors(profile_matrix1)  # Men find women

    profile_nn.fit(profile_matrix1)  # Fit on men
    dist2, idx2 = profile_nn.kneighbors(profile_matrix2)  # Women find men

    # ADD feature for profile matches
    df_1['profile_matches'] = list(idx1)
    df_2['profile_matches'] = list(idx2)

    # ADD feature for combined recommendation
    df_1['recommended_matches'] = df_1.apply(
        lambda row: combine_ranked_matches(
            essay_matches = row.essay_matches,
            profile_matches = row.profile_matches,
            essay_weight = 0.2,
            profile_weight = 0.8,
            final_n = 50
        ),
        axis=1
    )

    df_2['recommended_matches'] = df_2.apply(
        lambda row: combine_ranked_matches(
            essay_matches = row.essay_matches,
            profile_matches = row.profile_matches,
            essay_weight = 0.2,
            profile_weight = 0.8,
            final_n = 50
        ),
        axis=1
    )

    # ADD feature for combined recommendation user ids
    df_1['recommended_matches_user_ids'] = df_1.apply(
        lambda row: get_user_ids(
            recommendation_indices = row.recommended_matches,
            df_in = df_2
        ),
        axis=1
    )

    df_2['recommended_matches_user_ids'] = df_2.apply(
        lambda row: get_user_ids(
            recommendation_indices = row.recommended_matches,
            df_in = df_1
        ),
        axis=1
    )

    # Compare User at index (i) with their top 5 matches
    print('User Profile:')
    print(df_1.loc[i,(
        'user_id',
        'age',
        'body_type',
        'height',
        'smokes_yes',
        'smokes_no',
        'religion_affiliation',
        'last_online_priority'
    )].T)

    print(show_recommended_profiles(df_1, df_2, i, n=5).T)

    return df_1,df_2

