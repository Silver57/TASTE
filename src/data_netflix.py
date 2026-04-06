"""Netflix Prize data loading and filtering."""

import glob
import os

import pandas as pd


def parse_netflix_ratings(data_dir: str) -> pd.DataFrame:
    """Parse combined_data_*.txt files into a ratings DataFrame."""
    records = []
    current_movie_id = None

    for filepath in sorted(glob.glob(os.path.join(data_dir, "combined_data_*.txt"))):
        print(f"  Parsing {os.path.basename(filepath)} …")
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line.endswith(":"):
                    current_movie_id = int(line[:-1])
                else:
                    customer_id, rating, date = line.split(",")
                    records.append(
                        (current_movie_id, int(customer_id), int(rating), date)
                    )

    ratings = pd.DataFrame(records, columns=["movie_id", "customer_id", "rating", "date"])
    ratings["movie_id"] = ratings["movie_id"].astype("int16")
    ratings["customer_id"] = ratings["customer_id"].astype("int32")
    ratings["rating"] = ratings["rating"].astype("int8")
    ratings["date"] = pd.to_datetime(ratings["date"])
    return ratings


def load_netflix(data_dir: str, min_reviews: int = 20_000) -> pd.DataFrame:
    """
    Load Netflix data and return a unified DataFrame with columns:
    user_id, item_id, title, rating.

    Only movies with >= min_reviews are kept (proxy for model familiarity).
    """
    print("Loading Netflix data …")

    # Movie metadata
    with open(os.path.join(data_dir, "movie_titles.csv"), encoding="latin-1") as f:
        rows = [line.strip().split(",", 2) for line in f if line.strip()]
    movies = pd.DataFrame(rows, columns=["movie_id", "year", "title"])
    movies["movie_id"] = movies["movie_id"].astype("int16")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce").astype("Int16")

    # Ratings
    ratings = parse_netflix_ratings(data_dir)
    df = ratings.merge(movies, on="movie_id", how="left")

    print(f"  Raw: {len(df):,} ratings, {df['movie_id'].nunique():,} movies, "
          f"{df['customer_id'].nunique():,} users")

    # Filter to well-known movies
    counts = df.groupby("movie_id")["rating"].count()
    keep = counts[counts >= min_reviews].index
    df = df[df["movie_id"].isin(keep)].copy()

    print(f"  After filter (>={min_reviews:,} reviews): "
          f"{len(df):,} ratings, {df['movie_id'].nunique():,} movies")

    # Rename to unified schema
    df = df.rename(columns={"customer_id": "user_id", "movie_id": "item_id"})
    return df[["user_id", "item_id", "title", "rating"]]
