import sys, time, shutil, getpass, socket
import os

def backup_terminal_outputs(save_path):
    """
    Backup standard and error outputs in terminal to two .txt files named 'stdout.txt' and 'stderr.txt' 
    respectively real time. Terminal would still display outputs as usual.
    Args:
        save_path (directory): directory to save the two .txt files.
    """
    sys.stdout = SysStdLogger(os.path.join(save_path, 'stdout.txt'), sys.stdout)
    sys.stderr = SysStdLogger(os.path.join(save_path, 'stderr.txt'), sys.stderr)		
    
class SysStdLogger(object):
    def __init__(self, filename='terminal log.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
        self.log.write(''.join([time.strftime("%y-%m-%d %H:%M:%S"), '\n\n']))

    def write(self, message):
        if 'deprecated pixel format used' in message:
            pass
        else:
            self.terminal.write(message)
            self.log.write(message)

    def flush(self):
        pass

    def __del__(self):
        self.log.write(''.join(['\n', time.strftime("%y-%m-%d %H:%M:%S")]))
        self.log.close()

def backup_code(save_path, save_parent=False, ignored_in_current_folder=None, marked_in_parent_folder=None):
    """
    Backup files in current folder and parent folder to "[save_path]/backup_code".
    Also backup standard and error outputs in terminal.
    Args:
        save_path (directory): directory to backup code. If not specified, it would be set to './tmp/[%y%m%d_%H%M%S]'
        ignored_in_current_folder (list): files or folders in this list are ignored when copying files under current folder
        marked_in_parent_folder (list): folders in this list will be copied when copying files under parent folder 
    """
    if ignored_in_current_folder is None:
        ignored_in_current_folder = ['tmp', 'log', 'data', '__pycache__', 'output','sythc_data']
    if marked_in_parent_folder is None:
        marked_in_parent_folder = ['mylib']

    # create directory for backup code
    backup_code_dir = os.path.join(save_path, 'backup_code')
    os.makedirs(backup_code_dir)

    # backup important variables
    with open(os.path.join(backup_code_dir, 'CLI argument.txt'), 'w') as f:
        res = ''.join(['hostName: ', socket.gethostname(), '\n',
                    'account: ', getpass.getuser(), '\n',
                    'save_path: ', os.path.realpath(save_path), '\n', 
                    'CUDA_VISIBLE_DEVICES: ', str(os.environ.get('CUDA_VISIBLE_DEVICES')), '\n'])
        f.write(res)

        for i, _ in enumerate(sys.argv):
            f.write(sys.argv[i] + '\n')

    # copy current script additionally
    script_file = sys.argv[0]
    shutil.copy(script_file, backup_code_dir)

    # copy files in current folder
    current_folder_name = os.path.basename(sys.path[0])
    os.makedirs(os.path.join(backup_code_dir, current_folder_name))
    for file_path in os.listdir(sys.path[0]):
        if file_path not in ignored_in_current_folder:
            if os.path.isdir(file_path):
                shutil.copytree(os.path.join(sys.path[0], file_path), os.path.join(backup_code_dir, current_folder_name, file_path))
            elif os.path.isfile(file_path):
                shutil.copy(os.path.join(sys.path[0], file_path), os.path.join(backup_code_dir, current_folder_name))
            else:
                print('{} is a special file(socket, FIFO, device file) that would not be backup.'.format(file_path))

    # copy folders in parent folder
    if save_parent:
        os.makedirs(os.path.join(backup_code_dir, 'parent_folder_files'))
        for file_path in os.listdir('../'):
            if os.path.isdir(os.path.join(sys.path[0], '../', file_path)) and file_path in marked_in_parent_folder:
                shutil.copytree(os.path.join(sys.path[0], '../', file_path), os.path.join(backup_code_dir, file_path))
            elif os.path.isfile(os.path.join(sys.path[0], '../', file_path)):
                # print(os.path.join(sys.path[0], '../', file_path), os.path.join(backup_code_dir, 'parent_folder_files'))
                shutil.copy(os.path.join(sys.path[0], '../', file_path), os.path.join(backup_code_dir, 'parent_folder_files'))

import random
import numpy as np
import torch
def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else: 
        print('No seed is specified, therefore seed is collected from system clock.')