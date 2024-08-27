import torch 
import torchvision
import numpy as np
import pandas as pd
import os
import shutil

source_dir = 'D:\Projects\Dog Breed\Data\dogimages\Val'
target_dir = 'D:\Projects\Dog Breed\Data\dogimages'

    
# file_names = os.listdir(source_dir)
    
# for file_name in file_names:
#     shutil.move(os.path.join(source_dir, file_name), target_dir)

# def moveTree(sourceRoot, destRoot):
#     if not os.path.exists(destRoot):
#         return False
#     ok = True
#     for path, dirs, files in os.walk(sourceRoot):
#         relPath = os.path.relpath(path, sourceRoot)
#         destPath = os.path.join(destRoot, relPath)
#         if not os.path.exists(destPath):
#             os.makedirs(destPath)
#         for file in files:
#             destFile = os.path.join(destPath, file)
#             if os.path.isfile(destFile):
#                 print "Skipping existing file: " + os.path.join(relPath, file)
#                 ok = False
#                 continue
#             srcFile = os.path.join(path, file)
#             #print "rename", srcFile, destFile
#             os.rename(srcFile, destFile)
#     for path, dirs, files in os.walk(sourceRoot, False):
#         if len(files) == 0 and len(dirs) == 0:
#             os.rmdir(path)
#     return ok

# moveTree(source_dir, target_dir)

def merge_overwrite_Tree(sourceRoot, destRoot):
  #https://stackoverflow.com/questions/22588225/how-do-you-merge-two-directories-or-move-with-replace-from-the-windows-command
  '''
  Updates destenation root, overwrites existing files.
  :param sourceRoot: Root folder from wehere to copy the files
  :param destRoot: Destination folder where new folders and files are created and new files are added
  :return: !=0 in case of errors
  '''
  if not os.path.exists(destRoot):
    return 1
  ok = 0
  for path, dirs, files in os.walk(sourceRoot):
    relPath = os.path.relpath(path, sourceRoot)
    destPath = os.path.join(destRoot, relPath)
    if not os.path.exists(destPath):
      print("create: %s"%destPath)
      os.makedirs(destPath)
    for file in files:
      destFile = os.path.join(destPath, file)
      if os.path.isfile(destFile):
        print("\n...Will overwrite existing file: " + os.path.join(relPath, file))
        #ok = False
        #continue
      srcFile = os.path.join(path, file)
      # print "rename", srcFile, destFile
      # os.rename(srcFile, destFile) # performs move
      print("copy %s to %s"%(srcFile, destFile))
      shutil.copy(srcFile, destFile) # performs copy&overwrite
  return ok

merge_overwrite_Tree(source_dir, target_dir)