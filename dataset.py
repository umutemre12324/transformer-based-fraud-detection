import numpy as np
import pandas as pd

np.random.seed(42)

sequence_len = 20
num_feature = 5
n_samples_per_class = 500
total_samples = n_samples_per_class * 3

data = []
labels = []

# Normal kullanıcılar: 0
for _ in range(n_samples_per_class):
    click_count = np.random.normal(3, 1, sequence_len)
    page_duration = np.random.normal(5, 2, sequence_len)
    amount_spent = np.random.normal(10, 5, sequence_len)
    scroll_depth = np.random.normal(50, 15, sequence_len)
    cart_events = np.random.poisson(0.5, sequence_len)

    sample = np.stack([
        click_count, page_duration, amount_spent,
        scroll_depth, cart_events
    ])

    data.append(sample)
    labels.append(0)

for _ in range(n_samples_per_class):
    click_count = np.random.normal(15, 2, sequence_len)
    page_duration = np.random.normal(0.5, 0.2, sequence_len)
    amount_spent = np.random.normal(0, 0.1, sequence_len)
    scroll_depth = np.random.normal(80, 5, sequence_len)
    cart_events = np.random.poisson(0.1, sequence_len)

    sample = np.stack([
        click_count, page_duration, amount_spent,
        scroll_depth, cart_events
    ])

    data.append(sample)
    labels.append(1)

for _ in range(n_samples_per_class):
    click_count = np.random.normal(8, 3, sequence_len)
    page_duration = np.random.normal(2, 1, sequence_len)
    amount_spent = np.random.normal(100, 30, sequence_len)
    scroll_depth = np.random.normal(20, 10, sequence_len)
    cart_events = np.random.poisson(5, sequence_len)

    sample = np.stack([
        click_count, page_duration, amount_spent,
        scroll_depth, cart_events
    ])

    data.append(sample)
    labels.append(2)


X = np.array(data, dtype=np.float32)
y = np.array(labels, dtype=np.int64)


indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Sınıf dağılımları:")
print(pd.Series(y).value_counts().sort_index())


np.save("X.npy", X)
np.save("y.npy", y)

print("Veri seti oluşturuldu ve kaydedildi.")

# Veriyi DataFrame formatına dönüştür
X_flat = X.reshape(X.shape[0], -1)

columns = [
    f"{feature}_t{t}"
    for feature in [
        "click_count",
        "page_duration",
        "amount_spent",
        "scroll_depth",
        "cart_events"
    ]
    for t in range(sequence_len)
]

df = pd.DataFrame(X_flat, columns=columns)
df["label"] = y

# CSV olarak kaydet
df.to_csv("dataset.csv", index=False)

print("DataFrame boyutu:", df.shape)
print("Dataset.csv başarıyla kaydedildi.")
