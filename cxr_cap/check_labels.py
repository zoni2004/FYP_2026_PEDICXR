from dataset import build_binary_labels
df, _ = build_binary_labels("../data/image_labels_test.csv")
print("PNEUMONIA:")
print(df[df.label == 1].head(5))
print("NORMAL:")
print(df[df.label == 0].head(5))