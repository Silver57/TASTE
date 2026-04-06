"""GoodReads data loading."""

from pathlib import Path

import pandas as pd


def load_goodreads(data_dir: str) -> pd.DataFrame:
    """
    Load GoodReads data and return a unified DataFrame with columns:
    user_id, item_id, title, rating.
    """
    data_dir = Path(data_dir)
    print("Loading GoodReads data …")

    ratings = pd.read_csv(data_dir / "ratings.csv")
    books = pd.read_csv(data_dir / "books.csv")

    print(f"  ratings.csv: {len(ratings):,} rows")
    print(f"  books.csv:   {len(books):,} rows")

    df = ratings.merge(books[["book_id", "title"]], on="book_id", how="inner")

    print(f"  Merged: {len(df):,} ratings, {df['user_id'].nunique():,} users, "
          f"{df['book_id'].nunique():,} books")

    df = df.rename(columns={"book_id": "item_id"})
    return df[["user_id", "item_id", "title", "rating"]]
