from sklearn.model_selection import train_test_split

DATA = "hindidata/hindi_preprocessed.tsv"

with open(DATA, encoding="utf-8") as f:
    lines = f.readlines()

train, temp = train_test_split(lines, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

def save(name, data):
    with open(f"hindidata/{name}.tsv", "w", encoding="utf-8") as f:
        f.writelines(data)

save("train", train)
save("val", val)
save("test", test)

print("Split done")
