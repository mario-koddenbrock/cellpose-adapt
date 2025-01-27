import os


data_nuclei = [
    (
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/images_cropped_isotropic/20240305_P013T_B005_cropped_isotropic.tif",
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/labelmaps/Nuclei/20240305_P013T_B005_cropped_isotropic_nuclei-labels.tif",
    ),
    (
        "data/P013T/20240701_P013T_60xOil_Hoechst_SiRActin/images_cropped_isotropic/P013T_60xOil_Hoechst_SiRActin_C_004_cropped_isotropic.tif",
        "data/P013T/20240701_P013T_60xOil_Hoechst_SiRActin/labelmaps/Nuclei/20240701_P013T_C004_cropped_isotropic_nuclei-labels.tif",
    ),
    (
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/images_cropped_isotropic/20240305_P013T_A001a_cropped_isotropic.tif",
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/labelmaps/Nuclei/20240305_P013T_A001a_cropped_isotropic_nuclei-labels.tif",
    ),
    (
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/images_cropped_isotropic/20240305_P013T_A003_cropped_isotropic.tif",
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/labelmaps/Nuclei/20240305_P013T_A003_cropped_isotropic_nuclei-labels.tif",
    ),
    (
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/images_cropped_isotropic/20240305_P013T_A004_cropped_isotropic.tif",
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/labelmaps/Nuclei/20240305_P013T_A004_cropped_isotropic_nuclei-labels.tif",
    ),
    (
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/images_cropped_isotropic/20240305_P013T_A005_cropped_isotropic.tif",
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/labelmaps/Nuclei/20240305_P013T_A005_cropped_isotropic_nuclei-labels.tif",
    ),
    (
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/images_cropped_isotropic/20240305_P013T_A006_cropped_isotropic.tif",
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/labelmaps/Nuclei/20240305_P013T_A006_cropped_isotropic_nuclei-labels.tif",
    ),
    (
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/images_cropped_isotropic/20240305_P013T_B007a_cropped_isotropic.tif",
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/labelmaps/Nuclei/20240305_P013T_B007a_cropped_isotropic_nuclei-labels.tif",
    ),
    (
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/images_cropped_isotropic/20240305_P013T_D004_cropped_isotropic.tif",
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/labelmaps/Nuclei/20240305_P013T_D004_cropped_isotropic_nuclei-labels.tif",
    ),
]

data_membranes = [
    (
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/images_cropped_isotropic/20240305_P013T_A001a_cropped_isotropic.tif",
        "data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/labelmaps/Membranes/20240305_P013T_A001a_cropped_isotropic_membranes-labels.tif",
    ),
]

data = data_nuclei + data_membranes


if __name__ == "__main__":
    root = "../"

    for image_path, ground_truth_path in data_nuclei:
        if not os.path.exists(os.path.join(root, image_path)):
            print(f"Image file {image_path} does not exist.")
        if not os.path.exists(os.path.join(root, ground_truth_path)):
            print(f"Labels file {ground_truth_path} does not exist.")
