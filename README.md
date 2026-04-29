# Date-A-Scientist
***

## Notebook Overview
The `date-a-scientist.ipynb` notebook implements a comprehensive matchmaking system using the OKCupid sample dataset. <br>
It focuses on analyzing user profiles and essays to generate recommendations through several techniques:

- **Data Cleaning**:
    - Filters users based on activity (dropping inactive users) and location (targeting California-based users).
    - Handles missing values by filling categorical columns with "unknown".
    - Standardizes text by removing special characters and HTML.
- **Feature Engineering**:
    - Encodes user readiness to match and implements One-Hot Encoding (OHE) for categorical features like religion, drinking, and drug use.
    - Adds a `last_online_priority` weighting to favor recently active users.
- **TF-IDF Vectorization**:
    - Combines 10 distinct essay columns into a single document for each user.
    - Applies `TfidfVectorizer` to convert text into vectors, filtering out high-frequency stop words to focus on unique user expressions.
- **Cosine Similarity**:
    - Calculates the angular similarity between TF-IDF vectors to find users with the most similar essay content.
- **Implicit ALS (Alternating Least Squares)**:
    - Implements the `implicit` library's ALS algorithm to model user-item interactions.
    - Uses latent factors to provide recommendations based on the strength of connections in the interaction matrix.