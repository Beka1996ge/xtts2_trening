import os
import pandas as pd
import glob

# ფაილების მისამართები (შეცვალეთ რეალური მისამართები)
SOURCE_PATH = "/path/to/previous/project"
train_dir = os.path.join(SOURCE_PATH, "data/split/train")
val_dir = os.path.join(SOURCE_PATH, "data/split/val")

# ემბედინგების დირექტორია, თუ გვჭირდება
embeddings_dir = os.path.join(SOURCE_PATH, "data/embeddings/embeddings")

# მეტადატას წაკითხვა, თუ არსებობს წინა პროექტში
try:
    prev_metadata = pd.read_csv(os.path.join(SOURCE_PATH, "data/split/metadata.csv"))
    has_metadata = True
    print("ნაპოვნია არსებული მეტადატა!")
except:
    has_metadata = False
    print("არსებული მეტადატა ვერ მოიძებნა, ვქმნით ახალს ფაილების სახელების საფუძველზე")

# შევქმნათ ტრენინგის მეტადატა
train_files = glob.glob(os.path.join(train_dir, "*.wav"))
train_data = []

for audio_file in train_files:
    filename = os.path.basename(audio_file)
    
    if has_metadata:
        # თუ გვაქვს მეტადატა, ვიპოვოთ შესაბამისი ჩანაწერი
        file_id = os.path.splitext(filename)[0]
        meta_row = prev_metadata[prev_metadata['file_id'] == file_id]
        
        if not meta_row.empty:
            text = meta_row['text'].values[0]
        else:
            text = f"ტექსტი ფაილისთვის {filename}"  # პირობითი ტექსტი
    else:
        text = f"ტექსტი ფაილისთვის {filename}"  # პირობითი ტექსტი
    
    train_data.append({
        'audio_file': filename,
        'text': text,
        'speaker_id': 0,
        'language_id': 0
    })

# შევქმნათ ვალიდაციის მეტადატა
val_files = glob.glob(os.path.join(val_dir, "*.wav"))
val_data = []

for audio_file in val_files:
    filename = os.path.basename(audio_file)
    
    if has_metadata:
        # თუ გვაქვს მეტადატა, ვიპოვოთ შესაბამისი ჩანაწერი
        file_id = os.path.splitext(filename)[0]
        meta_row = prev_metadata[prev_metadata['file_id'] == file_id]
        
        if not meta_row.empty:
            text = meta_row['text'].values[0]
        else:
            text = f"ტექსტი ფაილისთვის {filename}"  # პირობითი ტექსტი
    else:
        text = f"ტექსტი ფაილისთვის {filename}"  # პირობითი ტექსტი
    
    val_data.append({
        'audio_file': filename,
        'text': text,
        'speaker_id': 0,
        'language_id': 0
    })

# შევქმნათ DataFrame-ები
train_df = pd.DataFrame(train_data)
val_df = pd.DataFrame(val_data)

# შევინახოთ ფაილები
train_df.to_csv("data/train_metadata.csv", index=False)
val_df.to_csv("data/val_metadata.csv", index=False)

print(f"შეიქმნა ტრენინგის მეტადატა {len(train_data)} ჩანაწერით")
print(f"შეიქმნა ვალიდაციის მეტადატა {len(val_data)} ჩანაწერით")

# ახლა გადაკოპირებულ აუდიო ფაილებს გადავიტანთ data/wavs/ დირექტორიაში
import shutil

# ტრენინგის ფაილები
for audio_file in train_files:
    filename = os.path.basename(audio_file)
    destination = os.path.join("data/wavs", filename)
    shutil.copy2(audio_file, destination)

# ვალიდაციის ფაილები
for audio_file in val_files:
    filename = os.path.basename(audio_file)
    destination = os.path.join("data/wavs", filename)
    shutil.copy2(audio_file, destination)

print(f"გადაკოპირდა {len(train_files) + len(val_files)} აუდიო ფაილი data/wavs/ დირექტორიაში")
