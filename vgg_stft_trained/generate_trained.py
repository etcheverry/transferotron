from threading import Thread
import sys
import subprocess
import os
import argparse

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('folder', metavar='folder', type=str, help='Path to the folder.')
parser.add_argument('content', metavar='content', type=str, help='Filename or "all".')
parser.add_argument('style', metavar='style', type=str, help='Filename or "all"')

args = parser.parse_args()
str_folder = args.folder
str_content = args.content
str_style = args.style

directory = str_folder + '/generated_trained/'
    
class TransferThread(Thread):

    def __init__(self, content, style, output):

        Thread.__init__(self)

        self.content = content
        self.style = style
        self.output = output

    def run(self):
        print('Starting thread : ' + self.output)

        if not os.path.exists(directory + '/' + self.output):
            os.makedirs(directory + self.output)
        fd = os.open(directory + self.output + '/' + self.output + '.log', os.O_RDWR|os.O_CREAT )
        os.dup2(fd, 1);
        filename = 'neural_style_transfer_sound_trained.py'
        subprocess.call([sys.executable, filename, self.content, self.style, directory + self.output + '/' + self.output])
        os.close(fd)
        print('Ending thread : ' + self.output)


files = []
for file in os.listdir(str_folder):
    if file.endswith(".wav"):
        files.append(os.path.join(str_folder, file))

        
        
if not os.path.exists(directory):
    os.makedirs(directory)

threads = []

def remove_extension(string):
    return os.path.splitext(os.path.split(string)[1])[0]
        
if(str_content == 'all' and str_style == 'all'):
    for f1 in files:
        for f2 in files:
            out_name = remove_extension(f1) + '_' + remove_extension(f2)
            threads.append(TransferThread(f1, f2, remove_extension(out_name)))
if(str_content != 'all' and str_style != 'all'):
    out_name = remove_extension(str_content) + '_' + remove_extension(str_style)
    threads.append(TransferThread(str_content, str_style, remove_extension(out_name)))
if(str_content == 'all' and str_style != 'all'):
    for f1 in files:
        out_name = remove_extension(f1) + '_' + remove_extension(str_style)
        threads.append(TransferThread(f1, str_style, remove_extension(out_name)))

if(str_content != 'all' and str_style == 'all'):
    for f2 in files:
        out_name = remove_extension(str_content) + '_' + remove_extension(f2)
        threads.append(TransferThread(str_content, f2, remove_extension(out_name)))

nb_max = 8
nb_threads = len(threads)
cpt = 0
while(cpt < nb_threads + 8):
    for t in range(nb_max):
        if(t + cpt < nb_threads):
            threads[t + cpt].start()

    if(t + cpt < nb_threads):
            threads[t + cpt].join()
    cpt = cpt + 8
