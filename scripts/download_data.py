"""
Download primary and secondary datasets from Hugging Face.

Primary: med2425/resume-job-fit-merged-v1 (93,733 examples)
Secondary: netsol/resume-score-details (1,031 examples, validation/reference only)
"""
import os
import sys

def download_datasets():
    from datasets import load_dataset
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    
    # --- Primary Dataset ---
    print("=" * 60)
    print("Downloading PRIMARY dataset: med2425/resume-job-fit-merged-v1")
    print("=" * 60)
    
    ds = load_dataset("med2425/resume-job-fit-merged-v1")
    
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
    
    print(f"Train split: {len(train_df)} rows")
    print(f"Test split:  {len(test_df)} rows")
    print(f"Total:       {len(train_df) + len(test_df)} rows")
    print(f"Columns:     {list(train_df.columns)}")
    print(f"\nLabel distribution (train):")
    print(train_df["label"].value_counts().to_string())
    print(f"\nLabel distribution (test):")
    print(test_df["label"].value_counts().to_string())
    
    train_path = os.path.join(data_dir, "resume_job_fit_train.parquet")
    test_path = os.path.join(data_dir, "resume_job_fit_test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    print(f"\nSaved: {train_path}")
    print(f"Saved: {test_path}")
    
    # --- Secondary Dataset ---
    print("\n" + "=" * 60)
    print("Downloading SECONDARY dataset: netsol/resume-score-details")
    print("(For external validation/reference only — NOT merged into training)")
    print("=" * 60)
    
    try:
        ds2 = load_dataset("netsol/resume-score-details")
        # This dataset may have a single split
        split_name = list(ds2.keys())[0]
        val_df = ds2[split_name].to_pandas()
        print(f"Split '{split_name}': {len(val_df)} rows")
        print(f"Columns: {list(val_df.columns)}")
        
        val_path = os.path.join(data_dir, "resume_score_details_validation.parquet")
        val_df.to_parquet(val_path, index=False)
        print(f"Saved: {val_path}")
    except Exception as e:
        print(f"[WARN] Could not download secondary dataset: {e}")
        print("Continuing without it — primary dataset is sufficient.")
    
    print("\n" + "=" * 60)
    print("Dataset download complete.")
    print("=" * 60)


if __name__ == "__main__":
    download_datasets()
