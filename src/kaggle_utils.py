# -*-coding:utf-8 -*-

import os
import zipfile


def dir2zip(kaggle_path='/kaggle/working/', file_dir='', download=True):
    zip_path = os.path.join(kaggle_path, file_dir + '.zip')
    zip = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)

    for path, dirNames, fileNames in os.walk(os.path.join(kaggle_path, file_dir)):
        fpath = path.replace(file_dir, '')
        for name in fileNames:
            # only save best checkpoint
            if '.pth' in name and 'best' not in name:
                continue
            fullName = os.path.join(path, name)
            name = fpath + '\\' + name
            zip.write(fullName, name)
    zip.close()

    if download:
        from IPython.display import FileLink
        return FileLink(zip_path)