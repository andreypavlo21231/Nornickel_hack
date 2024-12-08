import os
import shutil

def rename_files_in_directory(directory_path):
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    files.sort()
    
    for i, filename in enumerate(files, start=1):
        old_path = os.path.join(directory_path, filename)
        new_filename = f"{i}SYNT_V2{os.path.splitext(filename)[1]}" 
        new_path = os.path.join(directory_path, new_filename)
        os.rename(old_path, new_path)
        old_path = os.path.join(directory_path.replace('blurred','masks'), filename.replace('.jpg','.png'))
        new_filename = f"{i}SYNT_V2{os.path.splitext(filename)[1]}"
        new_path = os.path.join(directory_path.replace('blurred','masks'), new_filename)
        os.rename(old_path, new_path)
#пример использования
directory_path = r"D:\Nickel_2_0\synt_create\database\blurred"  #укажите путь к папке
rename_files_in_directory(directory_path)
