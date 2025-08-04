Young's Modulus Predictor (Material Informatics)



Project Overview:

Random Forest machine learning model that predicts the Young's Modulus of a theoretical isotropic material based on calculatable factors

Can help scientists and engineers screen theoretical materials for structural applications without using DFT, which is extremely computationally intensive



Data:

Extracts bulk and shear moduli and uses the Voigt-Reuss-Hill approximation to calculate the theoretical Young's Modulus of every material with an available bulk and shear modulus in Materials Project database

Extracts the debye temperature, density, band gap, volume, energy above hull, formation energy per atom, and its structure as features to predict the Young's Modulus

Uses pymatgen to calculate the avg mass, electronegativity, and atomic radius of the material to use in features as well

Cleans the dataset by eliminating any unstable or outlier materials



Model:

Used random forest regressor to predict the Young's Modulus 

Used brute force feature engineering to find the most accurate combination of feature selection



GUI:

Created a GUI that is connected to the most accurate models (R² > 0.9) with different features

Can select between the different models depending on information that is on hand

Enables real-time predictions for the Young's Modulus
