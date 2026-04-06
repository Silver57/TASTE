"""GoodReads data loading."""

from pathlib import Path

import pandas as pd


def load_goodreads(data_dir: str, min_reviews: int = 20_000) -> pd.DataFrame:
    """
    Load GoodReads data and return a unified DataFrame with columns:
    user_id, item_id, title, rating.

    Only books with >= min_reviews are kept (proxy for model familiarity).
    """
    data_dir = Path(data_dir)
    print("Loading GoodReads data …")

    ratings = pd.read_csv(data_dir / "ratings.csv")
    books = pd.read_csv(data_dir / "books.csv")

    print(f"  ratings.csv: {len(ratings):,} rows")
    print(f"  books.csv:   {len(books):,} rows")

    df = ratings.merge(books[["book_id", "title"]], on="book_id", how="inner")

    print(f"  Raw: {len(df):,} ratings, {df['book_id'].nunique():,} books, "
          f"{df['user_id'].nunique():,} users")

    # Filter to well-known books
    counts = df.groupby("book_id")["rating"].count()
    keep = counts[counts >= min_reviews].index
    df = df[df["book_id"].isin(keep)].copy()

    print(f"  After filter (>={min_reviews:,} reviews): "
          f"{len(df):,} ratings, {df['book_id'].nunique():,} books")

    df = df.rename(columns={"book_id": "item_id"})
    return df[["user_id", "item_id", "title", "rating"]]
