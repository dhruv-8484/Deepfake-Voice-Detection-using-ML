import os
import shutil

# ====== CHANGE THESE PATHS IF NEEDED ======
protocol_path = r"C:\Users\dhruv\Downloads\ASVspoof2019_LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
flac_folder = r"C:\Users\dhruv\Downloads\ASVspoof2019_LA\LA\ASVspoof2019_LA_train\flac"
project_dataset_path = r"C:\Users\dhruv\OneDrive\Desktop\FVD\dataset"
# ===========================================

real_folder = os.path.join(project_dataset_path, "real")
fake_folder = os.path.join(project_dataset_path, "fake")

os.makedirs(real_folder, exist_ok=True)
os.makedirs(fake_folder, exist_ok=True)

real_count = 0
fake_count = 0
limit = 200  # number of files per class

with open(protocol_path, "r") as file:
    for line in file:
        parts = line.strip().split()
        filename = parts[1] + ".flac"
        label = parts[-1]

        source_path = os.path.join(flac_folder, filename)

        if label == "bonafide" and real_count < limit:
            shutil.copy(source_path, real_folder)
            real_count += 1

        elif label == "spoof" and fake_count < limit:
            shutil.copy(source_path, fake_folder)
            fake_count += 1

        if real_count >= limit and fake_count >= limit:
            break

print("Done copying files!")
print(f"Real files copied: {real_count}")
print(f"Fake files copied: {fake_count}")
