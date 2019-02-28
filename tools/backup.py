import shutil

def backup_project_as_zip(project_dir, zip_file):
    """
    Creates Copy of Project Dir to Zip_file as a backup 
    """
    shutil.make_archive(zip_file.replace('.zip',''), 'zip', project_dir)
    pass
