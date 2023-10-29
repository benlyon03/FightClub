import os
import zipfile
import shutil

import numpy as np
#Given a path unzip a file of mp4s and organize them into a train, test and val folder

def extract_from_zip_path(directory_path):
  """ List the files in each class of the dataset given a URL with the zip file.

    Args:
      zip_url: A URL from which the files can be extracted from.

    Returns:
      List of files in each of the classes.
  """
  # Specify the path to your zip file and the destination directory
  zip_file = 'archive.zip'
  data_path = directory_path
# Create the destination directory if it doesn't exist
  if not os.path.exists(data_path):
    os.makedirs(data_path)

# Open the zip file
  with zipfile.ZipFile(os.path.join(directory_path,zip_file), 'r') as zip_ref:
    # Extract each file in the zip archive and place them in the destination directory
    for file_info in zip_ref.infolist():
      # print(file_info)
      # Construct the destination path with only the filename (no directories)
      destination_path = os.path.join(data_path, os.path.basename(file_info.filename))
      # Extract the file
      with open(destination_path, 'wb') as file:
            #Only extract MP4s
        if file_info.filename.endswith('.mp4'):
            file.write(zip_ref.read(file_info.filename))
  return os.listdir(data_path)

def moveData(data_dir):
  print(data_dir)
  #Create the folders
  if not os.path.exists(os.path.join(data_dir, 'train', 'NV')):
    os.makedirs(os.path.join(data_dir, 'train', 'NV'))
  if not os.path.exists(os.path.join(data_dir, 'train', 'V')):
    os.makedirs(os.path.join(data_dir, 'train', 'V'))
  if not os.path.exists(os.path.join(data_dir, 'test', 'NV')):
    os.makedirs(os.path.join(data_dir, 'test', 'NV'))
  if not os.path.exists(os.path.join(data_dir, 'test', 'V')):
    os.makedirs(os.path.join(data_dir, 'test', 'V'))
  if not os.path.exists(os.path.join(data_dir, 'val', 'NV')):
    os.makedirs(os.path.join(data_dir, 'val', 'NV'))
  if not os.path.exists(os.path.join(data_dir, 'val', 'V')):
    os.makedirs(os.path.join(data_dir, 'val', 'V'))

  #Get the list of files
  files = os.listdir(data_dir)

  #Get number of mp4 files
  mp4files = [file for file in files if file.endswith('.mp4')]
  nvfile = [file for file in mp4files if file.startswith('NV')]
  vfile = [file for file in mp4files if file.startswith('V')]

  #Make NV folder and V folder
  if not os.path.exists(os.path.join(data_dir, 'NV')):
    os.makedirs(os.path.join(data_dir, 'NV'))
  if not os.path.exists(os.path.join(data_dir, 'V')):
    os.makedirs(os.path.join(data_dir, 'V'))

  #Move NV files to NV folder
  for file in nvfile:
    shutil.move(os.path.join(data_dir, file), os.path.join(data_dir, 'NV'))
  #Move V files to V folder
  for file in vfile:
    shutil.move(os.path.join(data_dir, file), os.path.join(data_dir, 'V'))

def organize_data(main_dir, nv_dir, v_dir, split, nv_v_split):

  #Get the list of files
  nv_files = os.listdir(nv_dir)
  v_files = os.listdir(v_dir)

  #Shuffle the files
  np.random.shuffle(nv_files)
  np.random.shuffle(v_files)

  #Get number of files in each class
  nv_num = len(nv_files)
  v_num = len(v_files)

  #Get the number of files in each split
  nv_train_num = int(nv_num * split[0])
  nv_test_num = int(nv_num * split[1])
  nv_val_num = nv_num - nv_train_num - nv_test_num
  v_train_num = int(v_num * split[0])
  v_test_num = int(v_num * split[1])
  v_val_num = v_num - v_train_num - v_test_num

  #split list into train, test and val
  nv_train = nv_files[:nv_train_num]
  nv_test = nv_files[nv_train_num:nv_train_num + nv_test_num]
  nv_val = nv_files[nv_train_num + nv_test_num:]

  v_train = v_files[:v_train_num]
  v_test = v_files[v_train_num:v_train_num + v_test_num]
  v_val = v_files[v_train_num + v_test_num:]

  v_train = v_train[:int(len(nv_train) * nv_v_split[1])]
  v_test = v_test[:int(len(nv_test) * nv_v_split[1])]
  v_val = v_val[:int(len(nv_val) * nv_v_split[1])]
  #Move the files to the correct folder
  for file in nv_train:
    shutil.move(os.path.join(nv_dir, file), os.path.join(main_dir, 'train', 'NV'))
  for file in nv_test:
    shutil.move(os.path.join(nv_dir, file), os.path.join(main_dir, 'test', 'NV'))
  for file in nv_val:
    shutil.move(os.path.join(nv_dir, file), os.path.join(main_dir, 'val', 'NV'))

  for file in v_train:
    shutil.move(os.path.join(v_dir, file), os.path.join(main_dir, 'train', 'V'))
  for file in v_test:
    shutil.move(os.path.join(v_dir, file), os.path.join(main_dir, 'test', 'V'))
  for file in v_val:
    shutil.move(os.path.join(v_dir, file), os.path.join(main_dir, 'val', 'V'))

  #Remove V and NV folder
  shutil.rmtree(nv_dir)
  shutil.rmtree(v_dir)


#Add data folder if does not exist
if not os.path.exists('Data'):
  os.makedirs('Data')

#Move archive.zip to Data
shutil.move('archive.zip', 'Data')
extract_from_zip_path('Data')
moveData('Data')
#Call the function
organize_data('Data', 'Data/NV', 'Data/V', [0.6, 0.2, 0.2], [0.8, 0.2])

#Delete all mp4 file in Data and subfolders
for root, dirs, files in os.walk('Data'):
    for file in files:
        if file.endswith('.avi'):
            os.remove(os.path.join(root, file))

#Move archive.zip back to parent folder
shutil.move('Data/archive.zip', 'archive.zip')


#How many files are in each folder
print('Train NV: ', len(os.listdir('Data/train/NV')))
print('Train V: ', len(os.listdir('Data/train/V')))
print('Test NV: ', len(os.listdir('Data/test/NV')))
print('Test V: ', len(os.listdir('Data/test/V')))
print('Val NV: ', len(os.listdir('Data/val/NV')))
print('Val V: ', len(os.listdir('Data/val/V')))