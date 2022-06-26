from enum import auto
from heapq import merge
from weakref import ref
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook")

import larch
from larch import Group
from larch.xafs import autobk, xftf, mback,pre_edge, pre_edge_baseline, rebin_xafs
from larch.io import read_athena, read_ascii, create_athena, merge_groups
from xraydb import guess_edge, xray_edge

from pptemp import pptemp
from datetime import date


import glob
import re
import os

from PIL import Image

class larchppt(object):
    def __init__(self, *args):
        super(larchppt, self).__init__(*args)
        
        self.data = Group()
        self.transmission = Group()
        self.fluorescence = Group()
        self.reference = Group()
        self.set_data_type()
        self.init_pre_edge_kws()
        self.init_autobk_kws()

        self.init_group_list()
        
    def set_data_type(self, data_type="QAS"):
        
        if data_type in ["athena", "QAS", "ascii"]:
            self.data_type = data_type
        else:
            print("Data type much be chosen from athena, ascii, QAS")
    
    def read_data(self, path):
        
        if self.data_type == "athena":
            self.data = read_athena(path)
        elif self.data_type in ["QAS", "ascii"]:
            self.data = read_ascii(path)
    
    # Initialization and Setting parameters
    
    def init_pre_edge_kws(self):
        self.pre_edge_kws=dict(
                                  # Values are set to the default values used in Demeter
                                  pre1 = -150, # the lower bond of the function to fit the pre-edge
                                  pre2 = -30, # the upper bond of the function to fit the pre-edge
                                  norm1 = 50, # the lower bond of the function to fit the post-edge
                                  norm2= 2000, # the upper bond of the function to fit the post-edge
                                  nnorm= 2, # degree for the polynomials
                                  nvict= 0, # energy exponent to use for pre-edge fit.
                                )
    
    def set_pre_edge_kws(self, pre_edge_kws=None):
        if pre_edge_kws is not None:
            self.pre_edge_kws.update(pre_edge_kws)
    
    def init_group_list(self):
        self.group_list = []
        self.transmission_list = []
        self.fluorescence_list = []
        self.reference_list = []
        
    def add_group_list(self):
        self.transmission_list.append(self.transmission)
        self.fluorescence_list.append(self.fluorescence)
        self.reference_list.append(self.reference)
    
    def init_autobk_kws(self):
        self.autobk_kws=dict(                
                                rbkg = 1, # distance (in Ang) for chi(R) above which the signal is ignored. Default = 1.
                                nknots = None, # number of knots in spline.  If None, it will be determined. Don't use it.
                                kmin = 0, # minimum k value   [0]
                                kmax = None, # maximum k value   [full data range].
                                kweight = 2, # k weight for FFT.  Default value for larch is 1 but Athena uses 2
                                dk = 1, # FFT window window parameter.  Default value for larch is 0.1 but Athena uses 1.
                                win = "hanning", # FFT window function name.     ['hanning']
                                nfft = 2048, # array size to use for FFT [2048]
                                kstep = 0.05, # k step size to use for FFT [0.05]
                                k_std = None, # optional k array for standard chi(k).
                                chi_std = None, # optional chi array for standard chi(k).
                                nclamp = 3, # number of energy end-points for clamp [3]
                                clamp_lo = 0, # weight of low-energy clamp [0]
                                clamp_hi = 1, # weight of high-energy clamp [1]
                                calc_uncertainties = True, # Flag to calculate uncertainties in mu_0(E) and chi(k) [True]
                                err_sigma = 1 #sigma level for uncertainties in mu_0(E) and chi(k) [1]
                                )
        
    def set_autobk_kws(self, autobk_kws=None):
        if autobk_kws is not None:
            self.autobk_kws.update(autobk_kws)

    # Analysis

    def calc_mu(self):
        if self.data_type == "QAS":
            # Create groups
            self.transmission = Group()
            self.fluorescence = Group()
            self.reference = Group()
            
            # Copy header
            self.copy_header(self.transmission, self.data)
            self.copy_header(self.fluorescence, self.data)
            self.copy_header(self.reference, self.data)
            
            # Create transmission group
            self.transmission.title = "Transmission"
            self.transmission.filename = "Transmission"
            self.transmission.energy = self.data.energy
            self.transmission.mu = -np.log(self.data.it/self.data.i0)

            # Create fluorescence group
            self.fluorescence.title = "Fluorescence"
            self.fluorescence.filename = "Fluorescence"
            self.fluorescence.energy = self.data.energy
            self.fluorescence.mu = self.data.iff/self.data.i0
            
            # Create reference group
            self.reference.title = "Reference"
            self.reference.filename = "Reference"
            self.reference.energy = self.data.energy
            self.reference.mu = -np.log(self.data.ir/self.data.it)
        else:
            print("to be implemented")
    
    def pre_edge(self, group, pre_edge_kws=None, keep_e0 = False):
        if pre_edge_kws is not None:
            keywords = self.pre_edge_kws
            
            if keep_e0:
                keywords.update({"e0":group.e0})
                        
            keywords.update(pre_edge_kws)
            pre_edge(group, **keywords)

        else:
            pre_edge(group, **self.pre_edge_kws)
    
    def autobk(self, group, autobk_kws=None, pre_edge_kws=None):
        # pre_edge(group, **self.pre_edge_kws)
        
        if autobk_kws is not None:
            keywords = self.autobk_kws
            keywords.update(autobk_kws)
            autobk(group, **keywords)
        else:
            autobk(group, **self.autobk_kws)
    
    
    # Plotting
    
    def plot_mu_tfr(self, path=None, resize_factor = 1.0):
        fig, ax = plt.subplots()
        ax.plot(self.transmission.energy, self.transmission.mu, label='$x\mu_t$')
        ax.plot(self.fluorescence.energy, self.fluorescence.mu, label='$x\mu_f$')
        ax.plot(self.reference.energy, self.reference.mu, label='$x\mu_{ref}$')
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("$x\mu(E)$")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        if path is not None:
            fig.savefig(path, bbox_inches='tight')
            plt.close()
            self.resize_img(path, resize_factor)
        
    def plot_mu(self, group, plot_mu = "mu", plot_pre = False, plot_post = False, path=None, resize_factor = 1.0):
        fig, ax = plt.subplots()
        
        if plot_mu == "mu":    
            ax.plot(group.energy,group.mu)
            ax.set_ylabel("$x\mu(E)$")
            
            if plot_pre:
                ax.plot(group.energy,group.pre_edge)
            if plot_post:
                ax.plot(group.energy,group.post_edge)
                
        elif plot_mu =="norm":
            ax.plot(group.energy,group.norm)
            ax.set_ylabel("Normalized $x\mu(E)$")
        
        elif plot_mu =="flat":  
            ax.plot(group.energy,group.flat)
            ax.set_ylabel("Normalized $x\mu(E)$")

        ax.set_xlabel("Energy (eV)")
        
        if path is not None:
            fig.savefig(path, bbox_inches='tight')
            plt.close()
            
            self.resize_img(path, resize_factor)
    
    def plot_mu_list(self, group_list, plot_mu = "flat", path=None, resize_factor = 1.0, xlim = [-30, 50]):
        fig, ax = plt.subplots()
        
        if plot_mu == "mu":    
            ax.set_ylabel("$x\mu(E)$")
            
            for group in group_list:
                ax.plot(group.energy,group.mu)           
                
        elif plot_mu =="norm":
            ax.set_ylabel("Normalized $x\mu(E)$")
            
            for group in group_list:
                ax.plot(group.energy,group.norm)
        
        elif plot_mu =="flat":  
            ax.set_ylabel("Normalized $x\mu(E)$")
            
            for group in group_list:
                ax.plot(group.energy,group.flat)

        ax.set_xlabel("Energy (eV)")
        
        if xlim:
            ax.set_xlim(xlim + group_list[0].e0)
        
        if path is not None:
            fig.savefig(path, bbox_inches='tight')
            plt.close()
            
            self.resize_img(path, resize_factor)
    
    def plot_k(self, group, k_weight = 2, path=None, resize_factor = 1.0):
        fig, ax = plt.subplots()
                      
        ax.plot(group.k, group.chi*group.k**2)
        if k_weight == 1:
            ax.set_ylabel("$k\chi(k) \: (\mathrm{\AA}^{-1})$")
        elif k_weight > 1:
            ax.set_ylabel("$k^{"+str(k_weight)+"}\chi(k) \: (\mathrm{\AA}^{-"+str(k_weight)+"})$")
        
        ax.set_xlabel("Wavenumber ($\mathrm{\AA}^{-1}$)")
        
        if path is not None:
            fig.savefig(path, bbox_inches='tight')
            plt.close()
            
            self.resize_img(path, resize_factor)

    # Utility
    
    def resize_img(self, path, resize_factor = 1.0):
        
        if resize_factor != 1.0:
            img = Image.open(path)
            width, height = img.size
            
            newsize = (int(width*resize_factor), int(height*resize_factor))
            img = img.resize(newsize, Image.ANTIALIAS)
            img = img.save(path)

    def save_trf(self, path):
        project = create_athena(path)
        project.add_group(self.transmission)
        project.add_group(self.fluorescence)
        project.add_group(self.reference)
        project.save()

    def gen_plot_mu_trf(self,fig_dir , save_dir = None, name = "larch", resize_factor = 1.0, add_group = True, recaliberation=False, denergy = 0):
        self.calc_mu()
        self.autobk(self.transmission)
        self.autobk(self.fluorescence)
        self.autobk(self.reference)

        if recaliberation:
            # It is not working well
            self.recaliberation(denergy)
            
        if add_group:
            self.add_group_list()
            
        if save_dir is not None:
            self.save_trf(save_dir+name+".prj")
        
        self.plot_mu_tfr(path=fig_dir+"01_TRF Plot.png", resize_factor = resize_factor)
        
        self.plot_mu(self.transmission, plot_mu="mu", path=fig_dir+self.transmission.title+".png", resize_factor = resize_factor, plot_pre=True, plot_post=True)
        self.plot_mu(self.fluorescence, plot_mu="mu", path=fig_dir+self.fluorescence.title+".png", resize_factor = resize_factor, plot_pre=True, plot_post=True)
        self.plot_mu(self.reference, plot_mu="mu", path=fig_dir+self.reference.title+".png", resize_factor = resize_factor, plot_pre=True, plot_post=True)
        
        self.plot_mu(self.transmission, plot_mu="flat", path=fig_dir+self.transmission.title+" normalized.png", resize_factor = resize_factor)
        self.plot_mu(self.fluorescence, plot_mu="flat", path=fig_dir+self.fluorescence.title+" normalized.png", resize_factor = resize_factor)
        self.plot_mu(self.reference, plot_mu="flat", path=fig_dir+self.reference.title+" normalized.png", resize_factor = resize_factor)

    def gen_plot_mu_trf_list(self,fig_dir , save_dir = None, name = "larch", resize_factor = 1.0, add_group = False, recaliberation=False, denergy = 0, xlim=[-30, 50]):       
        self.plot_mu_list(self.transmission_list, plot_mu="flat", path=fig_dir+self.transmission.title+" all region.png", resize_factor = resize_factor, xlim=None)
        self.plot_mu_list(self.fluorescence_list, plot_mu="flat", path=fig_dir+self.fluorescence.title+" all region.png", resize_factor = resize_factor, xlim=None)
        self.plot_mu_list(self.reference_list, plot_mu="flat", path=fig_dir+self.reference.title+" all region.png", resize_factor = resize_factor, xlim=None)
        
        self.plot_mu_list(self.transmission_list, plot_mu="flat", path=fig_dir+self.transmission.title+" ({} to {}eV).png".format(*xlim), resize_factor = resize_factor, xlim = xlim)
        self.plot_mu_list(self.fluorescence_list, plot_mu="flat", path=fig_dir+self.fluorescence.title+" ({} to {}eV).png".format(*xlim), resize_factor = resize_factor, xlim = xlim)
        self.plot_mu_list(self.reference_list, plot_mu="flat", path=fig_dir+self.reference.title+" ({} to {}eV).png".format(*xlim), resize_factor = resize_factor, xlim = xlim)
  
    def gen_plot_summary(self,fig_dir , save_dir = None, name = "larch", resize_factor = 1.0, add_group = False, recaliberation=False, denergy = 0):
        self.calc_mu()
        self.autobk(self.transmission)
        self.autobk(self.fluorescence)
        self.autobk(self.reference)

        if recaliberation:
            # It is not working well
            self.recaliberation(denergy)
                        
        if save_dir is not None:
            self.save_trf(save_dir+name+".prj")
        
        self.plot_mu_tfr(path=fig_dir+"01_TRF Plot.png", resize_factor = resize_factor)
        
        self.plot_mu(self.transmission, plot_mu="flat", path=fig_dir+self.transmission.title+" normalized.png", resize_factor = resize_factor)
        self.plot_mu(self.fluorescence, plot_mu="flat", path=fig_dir+self.fluorescence.title+" normalized.png", resize_factor = resize_factor)
        self.plot_mu(self.reference, plot_mu="flat", path=fig_dir+self.reference.title+" normalized.png", resize_factor = resize_factor)
        
        self.plot_k(self.transmission, path=fig_dir+self.transmission.title+" k.png", resize_factor = resize_factor)
        self.plot_k(self.fluorescence, path=fig_dir+self.fluorescence.title+" k.png", resize_factor = resize_factor)
        self.plot_k(self.reference, path=fig_dir+self.reference.title+" k.png", resize_factor = resize_factor)

    def merge_groups(self):
        # Merge groups
        self.transmission = merge_groups(self.transmission_list, master=self.transmission_list[0], xarray='energy', 
                                         yarray='mu', kind='cubic', trim = True, calc_yerr=True)
        self.fluorescence = merge_groups(self.fluorescence_list, master=self.fluorescence_list[0], xarray='energy', 
                                    yarray='mu', kind='cubic', trim = True, calc_yerr=True)
        self.reference = merge_groups(self.reference_list, master=self.reference_list[0], xarray='energy', 
                            yarray='mu', kind='cubic', trim = True, calc_yerr=True)

        # Copy headers
        self.copy_header(self.transmission, self.transmission_list[0])
        self.copy_header(self.fluorescence, self.fluorescence_list[0])        
        self.copy_header(self.reference, self.reference_list[0])

        # pre_edge to calc e0
        self.pre_edge(self.transmission)
        self.pre_edge(self.fluorescence)
        self.pre_edge(self.reference)
                
        self.rebin()        
    
    def rebin(self):
        # rebin
        rebin_xafs(self.transmission.energy, mu = self.transmission.mu, group=self.transmission, e0=None, 
                   pre1=None, pre2=-30, pre_step=2, xanes_step=None, exafs1=50, exafs2=None, exafs_kstep=0.05, method='centroid')
        rebin_xafs(self.fluorescence.energy, mu = self.fluorescence.mu, group=self.fluorescence, e0=None, 
                   pre1=None, pre2=-30, pre_step=2, xanes_step=None, exafs1=50, exafs2=None, exafs_kstep=0.05, method='centroid')
        rebin_xafs(self.reference.energy, mu = self.reference.mu, group=self.reference, e0=None, 
                   pre1=None, pre2=-30, pre_step=2, xanes_step=None, exafs1=50, exafs2=None, exafs_kstep=0.05, method='centroid') 
               

    def recaliberation(self, denergy = 0):
                    
        if denergy == 0:
            edge = guess_edge(self.reference.e0)
            denergy = self.reference.e0 - xray_edge(*edge).energy
            
        self.reference.energy -= denergy
        self.transmission.energy -=  denergy
        self.fluorescence.energy -= denergy

        self.reference.e0 -= denergy
        self.transmission.energy -=  denergy
        self.fluorescence.energy -=  denergy
                                   
    
    def copy_header(self, group, master):
        
        try:
            group.header = master.header
        except:
            pass
        
        try:
            group.filename = master.filename
        except:
            pass
        
        try:
            group.path = master.path
        except:
            pass
        
        try:
            group.title = master.title
        except:
            pass

    # Automation
    
    def QAS_preanalysis(self, files_path, file_regex = re.compile(r".*[_/](.*)\.[a-zA-Z]+"), 
                        output_dir="./output/", resize_factor = 1.0, athena_output_dir = "./output/", recaliberation=False):
        """Automatic preanalysis of data collected in QAS Beamline

        Args:
            files (string): Give the path to files for glob function
                            Example: ./data/*.dat
        """

        files = glob.glob(files_path)
    
        
        for file in files:
            name = re.findall(file_regex, file)[0]
            
            self.read_data(file)
            
            save_dir = output_dir+name+"/"
            fig_dir = save_dir + "fig/"

            os.makedirs(fig_dir, exist_ok=True)

            self.gen_plot_mu_trf(fig_dir, save_dir=athena_output_dir, name=name, resize_factor=resize_factor)
        
        # Ouput of merge
                
        name = "all"
        save_dir = output_dir+name+"/"
        fig_dir = save_dir + "fig/"

        os.makedirs(fig_dir, exist_ok=True)

        self.gen_plot_mu_trf_list(fig_dir, save_dir=athena_output_dir, name=name, resize_factor=resize_factor)
        
        # Ouput of merge
        self.merge_groups()
        
        name = "merge"
        save_dir = output_dir+name+"/"
        fig_dir = save_dir + "fig/"

        os.makedirs(fig_dir, exist_ok=True)

        self.gen_plot_summary(fig_dir, save_dir=athena_output_dir, name=name, resize_factor=resize_factor)
        
        # self.generate_presenation(output_dir=output_dir, ppt_path=ppt_path)
        
    def generate_presenation(self, output_dir="./output/", ppt_path= "./output/preanalysis.pptx", dir_path = None):
        # initialization
        # presentation = pptemp.pptemp("./template.pptx")
        presentation = pptemp.pptemp()
            
        # Slide 1 Title
        slide = presentation.add_title_slide("Auto Preanalysis of QAS", str(date.today()))
            
        # Create slides from figures with label
        # Set use_bar=False if you don't want the bars to appear
        if dir_path == None:
            dir_path=output_dir + "/*/fig/"            
        
        presentation.add_figure_label_slide(dir_path=dir_path, dir_regex = re.compile(r".*[/_](.*)/.*/"))
        
        # save
        os.makedirs(os.path.dirname(ppt_path), exist_ok=True)
        presentation.save(ppt_path)  
        
def main():    
    lp = larchppt()
    
    lp.QAS_preanalysis(files_path="./*/data/*.dat", output_dir="./output/", athena_output_dir="./output/", recaliberation=True)
    lp.generate_presenation(output_dir="./output/", ppt_path="./output/preanalysis.pptx")
    
   
    
    

if __name__ == '__main__':
    main()
    

