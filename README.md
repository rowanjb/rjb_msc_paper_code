# Code relating to my MSc and the associated paper

This repo represents the majority of my MSc research and all code related to my ''main'' MSc paper.
It's cloned on both Graham/Nibi (Alliance Can) and on Wessex/Carthage (UAlberta HPC).
(Hence some functions are meant to be run on Graham and others on Carthage!)

--------

Directories:
--------
masks : contains nc files of various masks and meshes that are needed in the below scripts

Master list of files:
--------

🧐 AR7W_MLE_xsection : analyses of densities and s.f. across the AR7W section\
⚙️ Argo_gridding_ANHA4 : puts 1D Argo data from mixedlayer.ucsd.edu onto the ANHA4 grid\
🧐 energetics : analyses of EKE and energy transfer terms in ANHA4\
🏞️ figure_AR7W_MLE_xsection : produces figure of AR7W cross section showing the s.f. hard at work\
🏞️ figure_ConvR_ConvV.py : produces figure of convective resistance and volume in ANHA4\
🏞️ figure_LabSea_HC_and_SC.py : produces figure of heat and salt content in the Lab Sea\
🏞️ figure_LabSea_and_grid_map.py : produces figure of the Lab Sea and the ANHA4 grid\
🏞️ figure_MLDs_body.py : produces figure (maps) of MLDs in ANHA4 and LAB60\
🏞️ figure_MLDs_supplemental : produces figure (maps) of MLD anomalies compared to Argo\
🏞️ figure_MLE.py : produces figure showing the MLE ''magnitude'' in space and time\
🏞️ figure_biogeochem.py : produces time series of CO2 and oxygen contents from BLING\
🏞️ figure_energetics_supplemental : produces figure showing the energetics data from energetics.py\
🏞️ figure_ls3k_fluxes : produces figure of volume flux anomalies (1) per section and (2) over time series\
🏞️ filepaths.py : creates .txt's (saved outside this repo) of output files on Graham\
🧐 ls3k_biogeochem.py : analyses of CO2 and oxygen content in the interior Lab Sea over time\
🧐 ls3k_contents : analyses of heat and salt content in the interior Lab Sea over time\
🧐 ls3k_fluxes : analyses of boudnary -> interior volume, salt, heat, and freshwater fluxes\
⚙️ ls3k_mask_boundary.py : extracts the bounding cells defining the edge of the 3,000 m isobath mask\
⚙️ mask_maker.py : used to create the region masks used in the paper\
🧐 stratification.py : analyses relating to convective resistance, convective volume, and MLD for ANHA4, Argo, and LAB60

🧐 = analysis script using ANHA4, LAB60, and/or Argo data to produce .nc's used to produce figures \
⚙️ = utility script useful in some way for the analysis \
🏞️ = figure script using the .nc's from the analysis scripts to produce .svg's (which are touched up in Inkscape and then put into the paper)
