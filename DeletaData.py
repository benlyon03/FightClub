import os

#Delete all mp4 file in Data and subfolders
for root, dirs, files in os.walk('Data'):
    for file in files:
        if file.endswith('.mp4') or file.endswith('.avi'):
            os.remove(os.path.join(root, file))