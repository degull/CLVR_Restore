from data_loader import get_train_datasets

if __name__ == "__main__":
    datasets = get_train_datasets(limit=160)
    print(f"✅ Loaded datasets: {len(datasets)}")
    for i, d in enumerate(datasets):
        print(f"   ↳ Dataset {i} size: {len(d)}")

    if len(datasets) == 0:
        print("❌ ERROR: No datasets loaded. Check data paths!")
    else:
        total = sum(len(d) for d in datasets)
        print(f"✅ Total training samples: {total}")
