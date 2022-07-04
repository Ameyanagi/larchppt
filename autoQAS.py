
import glob
from importlib.resources import path
import os
import time
from turtle import width
from scipy.__config__ import show

from watchdog.events import RegexMatchingEventHandler
from watchdog.observers import Observer

import larchppt
import re
from IPython.display import clear_output
from PIL import Image
import numpy as np
import glob

from pyparsing import col
import matplotlib.pyplot as plt

class autoQAS(object):
    def __init__(self,
                 data_dir = "/nsls2/data/qas-new/legacy/processed/",
                 year = "2022",
                 cycle = "2",
                 proposal = "309890",
                 debug_dir = None,
                 *args):
        super(autoQAS, self).__init__(*args)
        self.data_dir = data_dir
        self.year = year
        self.cycle = cycle
        self.proposal = proposal
        self.debug_dir = debug_dir
        
        self.init_root_dir()
        self.update_watch_regex()
        self.update_name_regex()
        self.init_current_job_files()        
        self.lp = larchppt.larchppt()
        
    # Initialize root directory
    # If debug_dir is not None, use debug_dir instead of data_dir
    def init_root_dir(self):
        
        if self.debug_dir is None:      
            self.root_dir = self.data_dir + self.year + "/" + self.cycle + "/" + self.proposal + "/"
        else:
            self.root_dir = self.debug_dir
        
        if not os.path.exists(self.root_dir):
            print("Root directory does not exist: " + self.root_dir)
        
    def update_watch_regex(self, regex = [r'.*\.dat$']):
        self.watch_regex = regex
        
    def update_name_regex(self, regex = re.compile(r'([^/]*)  [0-9]{4}(?:-r[0-9]{4}\.dat)?')):
        self.name_regex = regex
    
    # Obtain current dat files in order of modification time
    def obtain_dat_list(self):
        list_of_dat_files = glob.glob(self.root_dir + "*.dat")
        self.dat_list = sorted( list_of_dat_files,
                        key = os.path.getmtime)
    
    def print_dat_list(self):
        print("\n".join(self.dat_list))
        
    def print_dat_list_length(self):
        print("Length of dat list: " + str(len(self.dat_list)))
        
    def print_dat_list_modified_time(self):
        for dat_file in self.dat_list:
            print(dat_file + ": " + time.ctime(os.path.getmtime(dat_file)))
    
    def on_modified(self, event):
        clear_output(wait=True)
        filepath = event.src_path
        print("File modified: " + filepath)
        self.update_current_job_files(filepath)
        self.QAS_preanalysis_update(filepath)
        
        self.gen_slides()
        
    def on_created(self, event):
        self.on_modified(event)
            
    def watch_update(self):
        event_handler = RegexMatchingEventHandler(self.watch_regex)
        event_handler.on_modified = self.on_modified
        event_handler.on_created = self.on_created
    
        observer = Observer()
        observer.schedule(event_handler, self.root_dir, recursive=True)
        observer.start()
        try:
            while True:
                time.sleep(10)
                
        except KeyboardInterrupt:
            observer.stop()
            
        observer.join()
        
    def obtain_name(self, file):
        return re.findall(self.name_regex, file)[0]
    
    def init_current_job_files(self):
        self.obtain_dat_list()
        self.get_current_job_files()
    
    def get_current_job_files(self):
        
        self.current_job_files = []
        
        if len(self.dat_list) == 0:
            return

        self.current_job_name = self.obtain_name(self.dat_list[-1])
        
        
        for file in self.dat_list:
            if self.obtain_name(file) == self.current_job_name:
                self.current_job_files.append(file)
    
    def update_current_job_files(self, path):
        
        if self.current_job_files == []:
            self.current_job_name = self.obtain_name(path)     
        
        if self.obtain_name(path) == self.current_job_name:
            self.current_job_files.append(path)
        else:
            self.current_job_name = self.obtain_name(path)
            self.current_job_files = [path]
        
    def QAS_preanalysis_update(self, file, file_regex=re.compile(r".*[_/](.*)\.[a-zA-Z]+"),
                        output_dir="../output/", resize_factor=1.0, recaliberation=False, fix_e0=False, prefix = "00000_", merge = True, show_img = True):
        
        current_job = self.obtain_name(file)
        name = re.findall(file_regex, file)[0]
                       
        save_dir = output_dir+"/"+self.current_job_name+"/"+name+"/"
        fig_dir = save_dir + "fig/"
        athena_output_dir = output_dir + "athena_each_spectrum/"
        
        
        
        os.makedirs(fig_dir, exist_ok=True)
        os.makedirs(athena_output_dir, exist_ok=True)
                
        if current_job != self.current_job_name:
            self.lp.init_group_list()
                
        self.lp.read_data(file)
        self.lp.gen_plot_mu_trf(fig_dir, save_dir=athena_output_dir, name=name, resize_factor=resize_factor, fix_e0=fix_e0)
        if show_img:
            self.show_img(fig_dir)

        if merge:
            # Ouput of merge

            if prefix: 
                name = "{} All Spectrum".format(prefix)
            else:
                name = current_job + " All Spectrum"
                
            save_dir = output_dir+"/"+self.current_job_name+"/"+name+"/"
            fig_dir = save_dir + "fig/"
            athena_output_dir = output_dir+ "athena_combined_spectrum/"

            os.makedirs(fig_dir, exist_ok=True)
            os.makedirs(athena_output_dir, exist_ok=True)

            self.lp.gen_plot_mu_trf_list(
                fig_dir, save_dir=athena_output_dir, name=name, resize_factor=resize_factor)
            if show_img:
                self.show_img(fig_dir)

            # Ouput of merge
            self.lp.merge_groups()

            if prefix: 
                name = "{} Merged Spectrum".format(prefix)
            else:
                name = current_job + " Merged Spectrum"
            save_dir = output_dir+"/"+self.current_job_name+"/"+name+"/"
            fig_dir = save_dir + "fig/"

            os.makedirs(fig_dir, exist_ok=True)

            self.lp.gen_plot_summary(
                fig_dir, save_dir=athena_output_dir, name=name, resize_factor=resize_factor)
            if show_img:
                self.show_img(fig_dir)

    def gen_slides(self):
        output_dir = "../output/{}/".format(self.current_job_name)
        title = "Preanalysis of {}".format(self.current_job_name)
        ppt_path = "../output/slides/{}_preanalysis.pptx".format(self.current_job_name)
        title_font_size = 20
        label_font_size = 12
        
        self.lp.generate_presenation(output_dir=output_dir, ppt_path=ppt_path, title = title, title_font_size=title_font_size, label_font_size=label_font_size)
    
    def show_current_img(self):
        output_dir = "../output/{}/".format(self.current_job_name)
        title = "Preanalysis of {}".format(self.current_job_name)
        ppt_path = "../output/slides/{}_preanalysis.pptx".format(self.current_job_name)
        title_font_size = 20
        label_font_size = 12
        
        self.lp.generate_presenation(output_dir=output_dir, ppt_path=ppt_path, title = title, title_font_size=title_font_size, label_font_size=label_font_size)
    
    
    def show_img(self, dir_path):
        path_list = glob.glob(dir_path + "/*.png")
        path_list.sort()
        
        # # Check path_list
        # print("Path list: ", path_list)
        
        img_length = len(path_list)
        
        columns = int(np.ceil(np.sqrt(img_length)))
        rows = int(np.ceil(img_length/columns))
        img = Image.open(path_list[0])
        width, height = img.size
        
        fig_height=columns*4
        fig_width=rows*6*width/height

        fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize = (fig_width, fig_height))

        for i, ax in enumerate(axes.flat, start=1):
            if (i > img_length):
                break
            ax.set_title(os.path.basename(path_list[i-1]))
            img = Image.open(path_list[i-1])
            ax.imshow(img)
            ax.set_axis_off()

        fig.tight_layout()

        plt.show()
               

    def watch_QAS(self):
        self.obtain_dat_list()
        self.get_current_job_files()
        
        for file in self.current_job_files:
            self.QAS_preanalysis_update(file, merge=False, show_img=False)
        
        self.gen_slides()       
        
        self.watch_update()
        
        
    def calc_align_img(self, path_list, left, top, width, height):
                
        columns = int(np.sqrt(len(path_list)))
        rows = np.ceil(len(path_list)/columns)
        
        width = width/rows
        height = height/columns
                
        results = []
                
        for i in range(len(path_list)):

            row = i%rows
            column = i//rows
            
            if column == len(path_list)//rows:
                residue = rows - len(path_list)%rows
            else:
                residue = 0
            
            results.append([path_list[i], left+width*(row+residue/2), top+height*column, width, height])
            
        return results