# This file is for testing only

from fileinput import filename
import larchppt

lp = larchppt.larchppt()

# analysis = "00"
# filename = "Co foil"
# lp.QAS_preanalysis(files_path="../analysis_Ryuichi/"+analysis+"*/data/*.dat", output_dir="../preanalysis/", 
#                    ppt_path="../preanalysis/"+filename+".pptx", resize_factor=1/2)

# just for generating the pptx
# lp.generate_presenation(output_dir="../preanalysis/", ppt_path="../preanalysis/"+filename+".pptx")

analysis_list = ["06", "07", "08", "09", "10", "11"]

for analysis in analysis_list:
    filename = "Co"+str(int(analysis))

    print(filename)
    # lp.QAS_preanalysis(files_path="../analysis_Ryuichi/"+analysis+"*/data/*.dat", output_dir="../preanalysis/", 
    #                    ppt_path="../preanalysis_slides/"+filename+".pptx", resize_factor=1/2)
    lp.generate_presenation(dir_path="../preanalysis/"+filename+"*/fig/", ppt_path="../preanalysis_slides/"+filename+".pptx")