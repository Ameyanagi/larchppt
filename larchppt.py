# module related to larch
import glob
import os
import re
from datetime import date

import matplotlib.pyplot as plt

# numpy, matplotlib
import numpy as np
import seaborn as sns
import tqdm
from larch import Group
from larch.io import create_athena, merge_groups, read_ascii, read_athena
from larch.xafs import autobk, pre_edge, rebin_xafs, xftf

# modle for utility functions
from PIL import Image
from xraydb import guess_edge, xray_edge

# module for ppt generation
import pptemp

sns.set_style("whitegrid")
sns.set_context("notebook")


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
        self.init_xftf_kws()
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
        self.pre_edge_kws = dict(
            # Values are set to the default values used in Demeter
            pre1=-150,  # the lower bond of the function to fit the pre-edge
            pre2=-50,  # the upper bond of the function to fit the pre-edge
            norm1=50,  # the lower bond of the function to fit the post-edge
            norm2=2000,  # the upper bond of the function to fit the post-edge
            nnorm=2,  # degree for the polynomials
            nvict=0,  # energy exponent to use for pre-edge fit.
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
        self.autobk_kws = dict(
            # distance (in Ang) for chi(R) above which the signal is ignored. Default = 1.
            rbkg=1,
            # number of knots in spline.  If None, it will be determined. Don't use it.
            nknots=None,
            kmin=0,  # minimum k value   [0]
            kmax=None,  # maximum k value   [full data range].
            kweight=2,  # k weight for FFT.  Default value for larch is 1 but Athena uses 2
            # FFT window window parameter.  Default value for larch is 0.1 but Athena uses 1.
            dk=1,
            win="hanning",  # FFT window function name.     ['hanning']
            nfft=2048,  # array size to use for FFT [2048]
            kstep=0.05,  # k step size to use for FFT [0.05]
            k_std=None,  # optional k array for standard chi(k).
            chi_std=None,  # optional chi array for standard chi(k).
            nclamp=3,  # number of energy end-points for clamp [3]
            clamp_lo=0,  # weight of low-energy clamp [0]
            clamp_hi=1,  # weight of high-energy clamp [1]
            # Flag to calculate uncertainties in mu_0(E) and chi(k) [True]
            calc_uncertainties=True,
            # sigma level for uncertainties in mu_0(E) and chi(k) [1]
            err_sigma=1,
        )

    def set_autobk_kws(self, autobk_kws=None):
        if autobk_kws is not None:
            self.autobk_kws.update(autobk_kws)

    def init_xftf_kws(self):
        self.xftf_kws = dict(
            rmax_out=10,  # highest R for output data (10 Ang)
            kweight=2,  # exponent for weighting spectra by k**kweight [2]
            kmin=2,  # starting k for FT Window
            kmax=15,  # ending k for FT Window
            dk=1,  # tapering parameter for FT Window
            dk2=None,  # second tapering parameter for FT Window
            window="hanning",  # name of window type
            nfft=2048,  # value to use for N_fft (2048).
            kstep=0.05,  # value to use for delta_k (0.05 Ang^-1).
            # output the phase as well as magnitude, real, imag  [False]
            with_phase=False,
        )

    def set_xftf_kws(self, xftf_kws=None):
        if xftf_kws is not None:
            self.xftf_kws.update(xftf_kws)

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
            self.transmission.mu = -np.log(self.data.it / self.data.i0)

            # Create fluorescence group
            self.fluorescence.title = "Fluorescence"
            self.fluorescence.filename = "Fluorescence"
            self.fluorescence.energy = self.data.energy
            self.fluorescence.mu = self.data.iff / self.data.i0

            # Create reference group
            self.reference.title = "Reference"
            self.reference.filename = "Reference"
            self.reference.energy = self.data.energy
            self.reference.mu = -np.log(self.data.ir / self.data.it)
        else:
            print("to be implemented")

    def pre_edge(self, group, pre_edge_kws=None, fix_e0=None):

        # pre_edge(group, **self.pre_edge_kws)
        pre_edge_keywords = self.pre_edge_kws

        if fix_e0 is not None:
            pre_edge_keywords["e0"] = fix_e0
        elif hasattr(group, "e0"):
            pre_edge_keywords["e0"] = group.e0

        if pre_edge_kws is not None:
            pre_edge_keywords.update(pre_edge_kws)
        pre_edge(group, **pre_edge_keywords)

    def autobk(self, group, autobk_kws=None, pre_edge_kws=None, fix_e0=None):
        # pre_edge(group, **self.pre_edge_kws)
        pre_edge_keywords = self.pre_edge_kws

        if fix_e0 is not None:
            pre_edge_keywords["e0"] = fix_e0
        elif hasattr(group, "e0"):
            pre_edge_keywords["e0"] = group.e0

        if pre_edge_kws is not None:
            pre_edge_keywords.update(pre_edge_kws)

        autobk_keywords = self.autobk_kws
        if autobk_kws is not None:
            autobk_keywords.update(autobk_kws)

        autobk(group, **autobk_keywords, pre_edge_kws=pre_edge_keywords)

    def xftf(self, group, xftf_kws=None):
        if xftf_kws is not None:
            keywords = self.xftf_kws

            keywords.update(xftf_kws)
            xftf(group, **keywords)

        else:
            xftf(group, **self.xftf_kws)

    # Plotting

    def plot_mu_tfr(self, path=None, resize_factor=1.0, error_bar=False):
        fig, ax = plt.subplots()
        ax.plot(self.transmission.energy, self.transmission.mu, label="$x\mu_t$")
        ax.plot(self.fluorescence.energy, self.fluorescence.mu, label="$x\mu_f$")
        ax.plot(self.reference.energy, self.reference.mu, label="$x\mu_{ref}$")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("$x\mu(E)$")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        if path is not None:
            fig.savefig(path, bbox_inches="tight")
            plt.close()
            self.resize_img(path, resize_factor)

    def plot_mu(
        self,
        group,
        plot_mu="mu",
        plot_pre=False,
        plot_post=False,
        path=None,
        resize_factor=1.0,
        xlim=False,
        e0=None,
        error_bar=False,
    ):
        fig, ax = plt.subplots()

        if plot_mu == "mu":
            ax.plot(group.energy, group.mu)
            ax.set_ylabel("$x\mu(E)$")

            if plot_pre:
                ax.plot(group.energy, group.pre_edge)
            if plot_post:
                ax.plot(group.energy, group.post_edge)

        elif plot_mu == "norm":
            ax.plot(group.energy, group.norm)
            ax.set_ylabel("Normalized $x\mu(E)$")

        elif plot_mu == "flat":
            ax.plot(group.energy, group.flat)
            ax.set_ylabel("Normalized $x\mu(E)$")

        ax.set_xlabel("Energy (eV)")

        if xlim:
            if e0 is not None:
                center = e0
            else:
                senter = group.e0
            ax.set_xlim(np.array(xlim) + center)

        if path is not None:
            fig.savefig(path, bbox_inches="tight")
            plt.close()

            self.resize_img(path, resize_factor)

    def plot_mu_list(
        self,
        group_list,
        plot_mu="flat",
        path=None,
        resize_factor=1.0,
        xlim=False,
        error_bar=False,
    ):
        fig, ax = plt.subplots()

        if plot_mu == "mu":
            ax.set_ylabel("$x\mu(E)$")

            for group in group_list:
                ax.plot(group.energy, group.mu)

        elif plot_mu == "norm":
            ax.set_ylabel("Normalized $x\mu(E)$")

            for group in group_list:
                ax.plot(group.energy, group.norm)

        elif plot_mu == "flat":
            ax.set_ylabel("Normalized $x\mu(E)$")

            for group in group_list:
                ax.plot(group.energy, group.flat)

        ax.set_xlabel("Energy (eV)")

        if xlim:
            ax.set_xlim(xlim + group_list[0].e0)

        if path is not None:
            fig.savefig(path, bbox_inches="tight")
            plt.close()

            self.resize_img(path, resize_factor)

    def plot_k(self, group, k_weight=2, path=None, resize_factor=1.0):
        fig, ax = plt.subplots()

        ax.plot(group.k, group.chi * group.k**2)
        if k_weight == 1:
            ax.set_ylabel("$k\chi(k) \: (\mathrm{\AA}^{-1})$")
        elif k_weight > 1:
            ax.set_ylabel(
                "$k^{"
                + str(k_weight)
                + "}\chi(k) \: (\mathrm{\AA}^{-"
                + str(k_weight)
                + "})$"
            )

        ax.set_xlabel("Wavenumber ($\mathrm{\AA}^{-1}$)")

        if path is not None:
            fig.savefig(path, bbox_inches="tight")
            plt.close()

            self.resize_img(path, resize_factor)

    def plot_k_list(self, group_list, k_weight=2, path=None, resize_factor=1.0):
        fig, ax = plt.subplots()

        for group in group_list:
            ax.plot(group.k, group.chi * group.k**2)
        if k_weight == 1:
            ax.set_ylabel("$k\chi(k) \: (\mathrm{\AA}^{-1})$")
        elif k_weight > 1:
            ax.set_ylabel(
                "$k^{"
                + str(k_weight)
                + "}\chi(k) \: (\mathrm{\AA}^{-"
                + str(k_weight)
                + "})$"
            )

        ax.set_xlabel("Wavenumber ($\mathrm{\AA}^{-1}$)")

        if path is not None:
            fig.savefig(path, bbox_inches="tight")
            plt.close()

            self.resize_img(path, resize_factor)

    def plot_R(self, group, k_weight=2, path=None, resize_factor=1.0):
        fig, ax = plt.subplots()

        ax.plot(group.r, group.chir_mag)
        ax.set_ylabel("$|\chi(k)| \: (\mathrm{\AA}^{" + str(-(k_weight + 1)) + "}$)")
        ax.set_xlabel("Radial distance ($\mathrm{\AA}$)")

        if path is not None:
            fig.savefig(path, bbox_inches="tight")
            plt.close()

            self.resize_img(path, resize_factor)

    def plot_R_list(self, group_list, k_weight=2, path=None, resize_factor=1.0):
        fig, ax = plt.subplots()

        for group in group_list:
            ax.plot(group.r, group.chir_mag)

        ax.set_ylabel("$|\chi(k)| \: (\mathrm{\AA}^{" + str(-(k_weight + 1)) + "}$)")
        ax.set_xlabel("Radial distance ($\mathrm{\AA}$)")

        if path is not None:
            fig.savefig(path, bbox_inches="tight")
            plt.close()

            self.resize_img(path, resize_factor)

    # Utility

    def resize_img(self, path, resize_factor=1.0):

        if resize_factor != 1.0:
            img = Image.open(path)
            width, height = img.size

            newsize = (int(width * resize_factor), int(height * resize_factor))
            img = img.resize(newsize, Image.ANTIALIAS)
            img = img.save(path)

    def save_trf(self, path):
        project = create_athena(path)
        project.add_group(self.transmission)
        project.add_group(self.fluorescence)
        project.add_group(self.reference)
        project.save()

    def save_group(self, path, mode="transmission"):
        if mode == "transmission":
            group = self.transmission
        elif mode == "fluorescence":
            group = self.fluorescence
        elif mode == "reference":
            group = self.reference
        else:
            raise ValueError(
                "mode must be one of 'transmission', 'fluorescence', 'reference'"
            )

        project = create_athena(path)
        project.add_group(group)
        project.save()

    def gen_plot_mu(
        self,
        fig_dir,
        save_dir=None,
        name="larch",
        resize_factor=1.0,
        add_group=True,
        recaliberation=False,
        denergy=0,
        xlim=[-30, 50],
        fix_e0=None,
        error_bar=False,
        mode="transmission",
        center_mode="guess",
    ):
        self.calc_mu()

        if mode == "transmission":
            group = self.transmission
            group_list = self.transmission_list
        elif mode == "fluorescence":
            group = self.fluorescence
            group_list = self.fluorescence_list
        elif mode == "reference":
            group = self.reference
            group_list = self.reference_list
        else:
            raise ValueError(
                "mode must be one of 'transmission', 'fluorescence', 'reference'"
            )

        group.e0 = None

        if fix_e0 is not None:
            add_group = True

            if len(group_list) > 0:
                group.e0 = group_list[0].e0

        self.autobk(group, fix_e0=group.e0)

        if recaliberation:
            # It is not working well.
            group(recaliberation=denergy)

        self.xftf(group)

        if center_mode == "guess":
            center = xray_edge(*guess_edge(group.e0)).energy

        if add_group:
            group_list.append(group)

        if save_dir is not None:
            os.makedirs(os.path.dirname(save_dir + name + ".prj"), exist_ok=True)
            self.save_group(save_dir + name + ".prj", mode=mode)

        filename = "{fig_dir}/{num:05}_{title} {mode}{discription}.png"
        filename_format = dict(
            fig_dir=fig_dir, num=0, title="", discription=" normalized", mode=mode
        )

        discription = ["", " normalized", " k", " R"]

        filename_format["title"] = group.title
        filename_format["num"] += 1
        filename_format["discription"] = discription[0]
        self.plot_mu(
            group,
            plot_mu="mu",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            plot_pre=True,
            plot_post=True,
            error_bar=error_bar,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[1]
        self.plot_mu(
            group,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            error_bar=error_bar,
        )

        filename_format["num"] += 1
        filename_format["discription"] = " ({} to {}eV)".format(*xlim)
        self.plot_mu(
            group,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            xlim=xlim,
            e0=center,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[2]
        self.plot_k(
            group, path=filename.format(**filename_format), resize_factor=resize_factor
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[3]
        self.plot_R(
            group, path=filename.format(**filename_format), resize_factor=resize_factor
        )

    def gen_plot_mu_trf(
        self,
        fig_dir,
        save_dir=None,
        name="larch",
        resize_factor=1.0,
        add_group=True,
        recaliberation=False,
        denergy=0,
        xlim=[-30, 50],
        fix_e0=None,
        error_bar=False,
    ):
        self.calc_mu()

        self.transmission.e0 = None
        self.fluorescence.e0 = None
        self.reference.e0 = None

        if fix_e0 is not None:
            add_group = True

            if len(self.transmission_list) > 0:
                self.transmission.e0 = self.transmission_list[0].e0
            if len(self.fluorescence_list) > 0:
                self.fluorescence.e0 = self.fluorescence_list[0].e0
            if len(self.transmission_list) > 0:
                self.reference.e0 = self.reference_list[0].e0

        self.autobk(self.transmission, fix_e0=self.transmission.e0)
        self.autobk(self.fluorescence, fix_e0=self.fluorescence.e0)
        self.autobk(self.reference, fix_e0=self.reference.e0)

        if recaliberation:
            # It is not working well
            self.recaliberation(denergy)

        self.xftf(self.transmission)
        self.xftf(self.fluorescence)
        self.xftf(self.reference)

        center = xray_edge(*guess_edge(self.reference.e0)).energy

        if add_group:
            self.add_group_list()

        if save_dir is not None:
            os.makedirs(os.path.dirname(save_dir + name + ".prj"), exist_ok=True)
            self.save_trf(save_dir + name + ".prj")

        filename = "{fig_dir}/{num:05}_{title}{discription}.png"
        filename_format = dict(
            fig_dir=fig_dir, num=0, title="", discription=" normalized"
        )

        discription = ["", " normalized", " k", " R"]

        # # TRF Plot
        # filename_format["title"] = "TRF Plot"
        # filename_format["num"] += 1
        # filename_format["discription"] = discription[0]
        # self.plot_mu_tfr(path=filename.format(
        #     **filename_format), resize_factor=resize_factor)

        # Transmission Plot
        group = self.transmission

        filename_format["title"] = group.title
        filename_format["num"] += 1
        filename_format["discription"] = discription[0]
        self.plot_mu(
            group,
            plot_mu="mu",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            plot_pre=True,
            plot_post=True,
            error_bar=error_bar,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[1]
        self.plot_mu(
            group,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            error_bar=error_bar,
        )

        filename_format["num"] += 1
        filename_format["discription"] = " ({} to {}eV)".format(*xlim)
        self.plot_mu(
            group,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            xlim=xlim,
            e0=center,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[2]
        self.plot_k(
            group, path=filename.format(**filename_format), resize_factor=resize_factor
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[3]
        self.plot_R(
            group, path=filename.format(**filename_format), resize_factor=resize_factor
        )

        # Fluorescence Plot
        group = self.fluorescence

        filename_format["title"] = group.title
        filename_format["num"] += 1
        filename_format["discription"] = discription[0]
        self.plot_mu(
            group,
            plot_mu="mu",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            plot_pre=True,
            plot_post=True,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[1]
        self.plot_mu(
            group,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
        )

        filename_format["num"] += 1
        filename_format["discription"] = " ({} to {}eV)".format(*xlim)
        self.plot_mu(
            group,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            xlim=xlim,
            e0=center,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[2]
        self.plot_k(
            group, path=filename.format(**filename_format), resize_factor=resize_factor
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[3]
        self.plot_R(
            group, path=filename.format(**filename_format), resize_factor=resize_factor
        )

        # Reference Plot
        group = self.reference

        filename_format["title"] = group.title
        filename_format["num"] += 1
        filename_format["discription"] = discription[0]
        self.plot_mu(
            group,
            plot_mu="mu",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            plot_pre=True,
            plot_post=True,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[1]
        self.plot_mu(
            group,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
        )

        filename_format["num"] += 1
        filename_format["discription"] = " ({} to {}eV)".format(*xlim)
        self.plot_mu(
            group,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            xlim=xlim,
            e0=center,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[2]
        self.plot_k(
            group, path=filename.format(**filename_format), resize_factor=resize_factor
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[3]
        self.plot_R(
            group, path=filename.format(**filename_format), resize_factor=resize_factor
        )

    def gen_plot_mu_trf_list(
        self,
        fig_dir,
        save_dir=None,
        name="larch",
        resize_factor=1.0,
        add_group=False,
        recaliberation=False,
        denergy=0,
        xlim=[-30, 50],
    ):

        filename = "{fig_dir}/{num:05}_{title}{discription}.png"
        filename_format = dict(
            fig_dir=fig_dir, num=0, title="", discription=" normalized"
        )

        discription = ["", " all region", " k(all)", " R(all)"]

        # Transmission Plot
        group_list = self.transmission_list

        filename_format["title"] = group_list[0].title
        filename_format["num"] += 1
        filename_format["discription"] = discription[1]
        self.plot_mu_list(
            group_list,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            xlim=None,
        )

        filename_format["num"] += 1
        filename_format["discription"] = " ({} to {}eV)".format(*xlim)
        self.plot_mu_list(
            group_list,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            xlim=xlim,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[2]
        self.plot_k_list(
            group_list,
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[3]
        self.plot_R_list(
            group_list,
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
        )

        # Fluorescence Plot
        group_list = self.fluorescence_list

        filename_format["title"] = group_list[0].title
        filename_format["num"] += 1
        filename_format["discription"] = discription[1]
        self.plot_mu_list(
            group_list,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            xlim=None,
        )

        filename_format["num"] += 1
        filename_format["discription"] = " ({} to {}eV)".format(*xlim)
        self.plot_mu_list(
            group_list,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            xlim=xlim,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[2]
        self.plot_k_list(
            group_list,
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[3]
        self.plot_R_list(
            group_list,
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
        )

        # Reference Plot
        group_list = self.reference_list

        filename_format["title"] = group_list[0].title
        filename_format["num"] += 1
        filename_format["discription"] = discription[1]
        self.plot_mu_list(
            group_list,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            xlim=None,
        )

        filename_format["num"] += 1
        filename_format["discription"] = " ({} to {}eV)".format(*xlim)
        self.plot_mu_list(
            group_list,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            xlim=xlim,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[2]
        self.plot_k_list(
            group_list,
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[3]
        self.plot_R_list(
            group_list,
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
        )

    def gen_plot_summary(
        self,
        fig_dir,
        save_dir=None,
        name="larch",
        resize_factor=1.0,
        add_group=False,
        recaliberation=False,
        denergy=0,
        error_bar=False,
    ):
        self.calc_mu()
        self.autobk(self.transmission)
        self.autobk(self.fluorescence)
        self.autobk(self.reference)

        self.xftf(self.transmission)
        self.xftf(self.fluorescence)
        self.xftf(self.reference)

        if recaliberation:
            # It is not working well
            self.recaliberation(denergy)

        if save_dir is not None:
            self.save_trf(save_dir + name + ".prj")

        filename = "{fig_dir}/{num:05}_{title}{discription}.png"
        filename_format = dict(
            fig_dir=fig_dir, num=0, title="", discription=" normalized"
        )

        discription = ["", " normalized", " k", " R"]

        # TRF Plot
        filename_format["title"] = "TRF Plot"
        filename_format["num"] += 1
        filename_format["discription"] = discription[0]
        self.plot_mu_tfr(
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            error_bar=error_bar,
        )

        # Transmission Plot
        group = self.transmission
        filename_format["title"] = group.title
        filename_format["num"] += 1
        filename_format["discription"] = discription[1]
        self.plot_mu(
            group,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            error_bar=error_bar,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[2]
        self.plot_k(
            group, path=filename.format(**filename_format), resize_factor=resize_factor
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[3]
        self.plot_R(
            group, path=filename.format(**filename_format), resize_factor=resize_factor
        )

        # Fluorescence Plot
        group = self.fluorescence
        filename_format["title"] = group.title
        filename_format["num"] += 1
        filename_format["discription"] = discription[1]
        self.plot_mu(
            group,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            error_bar=error_bar,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[2]
        self.plot_k(
            group, path=filename.format(**filename_format), resize_factor=resize_factor
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[3]
        self.plot_R(
            group, path=filename.format(**filename_format), resize_factor=resize_factor
        )

        # Reference Plot
        group = self.reference
        filename_format["title"] = group.title
        filename_format["num"] += 1
        filename_format["discription"] = discription[1]
        self.plot_mu(
            group,
            plot_mu="flat",
            path=filename.format(**filename_format),
            resize_factor=resize_factor,
            error_bar=error_bar,
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[2]
        self.plot_k(
            group, path=filename.format(**filename_format), resize_factor=resize_factor
        )

        filename_format["num"] += 1
        filename_format["discription"] = discription[3]
        self.plot_R(
            group, path=filename.format(**filename_format), resize_factor=resize_factor
        )

    def merge_group(self, mode="transmission"):

        if mode == "transmission":
            group = self.transmission
            group_list = self.transmission_list
        elif mode == "fluorescence":
            group = self.fluorescence
            group_list = self.fluorescence_list
        elif mode == "reference":
            group = self.reference
            group_list = self.reference_list
        else:
            raise ValueError(
                "mode must be 'transmission', 'fluorescence' or 'reference'"
            )

        group = group.merge(
            group_list,
            master=group_list[0],
            xarray="energy",
            yarray="mu",
            kind="cubic",
            trim=True,
            calc_yerr=True,
        )

        self.copy_header(group, group_list[0])
        self.pre_edge(group)

        self.rebin(mode=mode)

    def merge_groups(self):
        # Merge groups
        self.transmission = merge_groups(
            self.transmission_list,
            master=self.transmission_list[0],
            xarray="energy",
            yarray="mu",
            kind="cubic",
            trim=True,
            calc_yerr=True,
        )
        self.fluorescence = merge_groups(
            self.fluorescence_list,
            master=self.fluorescence_list[0],
            xarray="energy",
            yarray="mu",
            kind="cubic",
            trim=True,
            calc_yerr=True,
        )
        self.reference = merge_groups(
            self.reference_list,
            master=self.reference_list[0],
            xarray="energy",
            yarray="mu",
            kind="cubic",
            trim=True,
            calc_yerr=True,
        )

        # Copy headers
        self.copy_header(self.transmission, self.transmission_list[0])
        self.copy_header(self.fluorescence, self.fluorescence_list[0])
        self.copy_header(self.reference, self.reference_list[0])

        # pre_edge to calc e0
        self.pre_edge(self.transmission)
        self.pre_edge(self.fluorescence)
        self.pre_edge(self.reference)

        self.rebin()

    def rebin(self, mode="trf"):
        if mode == "trf":
            rebin_xafs(
                self.transmission.energy,
                mu=self.transmission.mu,
                group=self.transmission,
                e0=None,
                pre1=None,
                pre2=-30,
                pre_step=2,
                xanes_step=None,
                exafs1=50,
                exafs2=None,
                exafs_kstep=0.05,
                method="centroid",
            )
            rebin_xafs(
                self.fluorescence.energy,
                mu=self.fluorescence.mu,
                group=self.fluorescence,
                e0=None,
                pre1=None,
                pre2=-30,
                pre_step=2,
                xanes_step=None,
                exafs1=50,
                exafs2=None,
                exafs_kstep=0.05,
                method="centroid",
            )
            rebin_xafs(
                self.reference.energy,
                mu=self.reference.mu,
                group=self.reference,
                e0=None,
                pre1=None,
                pre2=-30,
                pre_step=2,
                xanes_step=None,
                exafs1=50,
                exafs2=None,
                exafs_kstep=0.05,
                method="centroid",
            )
        elif mode == "transmission":
            rebin_xafs(
                self.transmission.energy,
                mu=self.transmission.mu,
                group=self.transmission,
                e0=None,
                pre1=None,
                pre2=-30,
                pre_step=2,
                xanes_step=None,
                exafs1=50,
                exafs2=None,
                exafs_kstep=0.05,
                method="centroid",
            )
        elif mode == "fluorescence":
            rebin_xafs(
                self.fluorescence.energy,
                mu=self.fluorescence.mu,
                group=self.fluorescence,
                e0=None,
                pre1=None,
                pre2=-30,
                pre_step=2,
                xanes_step=None,
                exafs1=50,
                exafs2=None,
                exafs_kstep=0.05,
                method="centroid",
            )
        elif mode == "reference":
            rebin_xafs(
                self.reference.energy,
                mu=self.reference.mu,
                group=self.reference,
                e0=None,
                pre1=None,
                pre2=-30,
                pre_step=2,
                xanes_step=None,
                exafs1=50,
                exafs2=None,
                exafs_kstep=0.05,
                method="centroid",
            )
        else:
            raise ValueError(
                "mode must be 'trf', 'transmission', 'fluorescence' or 'reference'"
            )

    def recaliberation(self, denergy=0):

        if denergy == 0:
            edge = guess_edge(self.reference.e0)
            denergy = self.reference.e0 - xray_edge(*edge).energy

        self.reference.energy -= denergy
        self.transmission.energy -= denergy
        self.fluorescence.energy -= denergy

        self.reference.e0 -= denergy
        self.transmission.energy -= denergy
        self.fluorescence.energy -= denergy

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

    def QAS_preanalysis(
        self,
        files_path,
        file_regex=re.compile(r".*[_/](.*)\.[a-zA-Z]+"),
        output_dir="./output/",
        resize_factor=1.0,
        athena_output_dir="./output/",
        recaliberation=False,
        fix_e0=False,
        prefix=None,
    ):
        """Automatic preanalysis of data collected in QAS Beamline

        Args:
            files (string): Give the path to files for glob function
                            Example: ./data/*.dat
        """

        files = glob.glob(files_path)
        files.sort()
        self.init_group_list()

        for file in tqdm.tqdm(files):
            name = re.findall(file_regex, file)[0]

            self.read_data(file)

            save_dir = output_dir + name + "/"
            fig_dir = save_dir + "fig/"

            os.makedirs(fig_dir, exist_ok=True)

            try:
                self.gen_plot_mu(
                    fig_dir,
                    save_dir=athena_output_dir,
                    name=name,
                    resize_factor=resize_factor,
                    fix_e0=fix_e0,
                    mode="transmission",
                )
            except:
                print("Error in processing transmittion file:", file)

            try:
                self.gen_plot_mu(
                    fig_dir,
                    save_dir=athena_output_dir,
                    name=name,
                    resize_factor=resize_factor,
                    fix_e0=fix_e0,
                    mode="fluorescence",
                )
            except:
                print("Error in processing in fluorescence. File: ", file)

            try:
                self.gen_plot_mu(
                    fig_dir,
                    save_dir=athena_output_dir,
                    name=name,
                    resize_factor=resize_factor,
                    fix_e0=fix_e0,
                    mode="reference",
                )
            except:
                print("Error in processing in reference. File: ", file)

        # Ouput of merge

        if prefix:
            name = "{} All Spectrum".format(prefix)
        else:
            name = "All Spectrum"
        save_dir = output_dir + name + "/"
        fig_dir = save_dir + "fig/"

        os.makedirs(fig_dir, exist_ok=True)

        try:
            self.gen_plot_mu_trf_list(
                fig_dir,
                save_dir=athena_output_dir,
                name=name,
                resize_factor=resize_factor,
            )
        except:
            print("Error in plotting all spectrum. File: ", name)

        # Ouput of merge
        try:
            self.merge_groups()

            if prefix:
                name = "{} Merged Spectrum".format(prefix)
            else:
                name = "Merged Spectrum"
            save_dir = output_dir + name + "/"
            fig_dir = save_dir + "fig/"

            os.makedirs(fig_dir, exist_ok=True)

            self.gen_plot_summary(
                fig_dir,
                save_dir=athena_output_dir,
                name=name,
                resize_factor=resize_factor,
            )
        except:
            print("Error in plotting merged spectrum. File: ", name)

        # self.generate_presenation(output_dir=output_dir, ppt_path=ppt_path)

    def generate_presenation(
        self,
        output_dir="./output/",
        ppt_path="./output/preanalysis.pptx",
        dir_path=None,
        title_font_size=30,
        title="Auto Preanalysis of QAS",
        label_font_size=18,
    ):
        # initialization
        # presentation = pptemp.pptemp("./template.pptx")
        presentation = pptemp.pptemp()

        # Slide 1 Title
        slide = presentation.add_title_slide(title, str(date.today()))

        # Create slides from figures with label
        # Set use_bar=False if you don't want the bars to appear
        if dir_path == None:
            dir_path = output_dir + "/*/fig/"

        presentation.add_figure_label_slide(
            dir_path=dir_path,
            dir_regex=re.compile(r".*[/_](.*)/.*/"),
            title_font_size=title_font_size,
            label_font_size=label_font_size,
        )

        # save
        os.makedirs(os.path.dirname(ppt_path), exist_ok=True)
        presentation.save(ppt_path)


def main():
    lp = larchppt()

    prefix_list = ["Co1", "Co11"]

    for prefix in prefix_list:
        lp.QAS_preanalysis(
            files_path=f"./{prefix}/data/*.dat",
            output_dir=f"./output/{prefix}/",
            athena_output_dir=f"./output/{prefix}/",
            fix_e0=True,
            prefix=f"00000_{prefix}",
        )

    lp.generate_presenation(
        output_dir="./output/Co*/",
        ppt_path="./output/Co_preanalysis.pptx",
        title_font_size=20,
        title="Preanalysis of Co samples",
    )


if __name__ == "__main__":
    main()
