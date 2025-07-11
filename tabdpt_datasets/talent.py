import os
import zipfile
from pathlib import Path

import gdown
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from tabdpt_datasets.dataset import Dataset


class TalentDataset(Dataset):
    """
    Dataset for LAMDA-TALENT baseline
    """

    @staticmethod
    def suite_name():
        return "talent"

    @staticmethod
    def all_names():
        return [
            "1000-Cameras-Dataset",
            "2dplanes",
            "3D_Estimation_using_RSSI_of_WLAN_dataset",
            "3D_Estimation_using_RSSI_of_WLAN_dataset_complete_1_target",
            "abalone",
            "Abalone_reg",
            "accelerometer",
            "ada",
            "ada_agnostic",
            "ada_prior",
            "Ailerons",
            "airfoil_self_noise",
            "airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True",
            "allbp",
            "allrep",
            "Amazon_employee_access",
            "analcatdata_authorship",
            "analcatdata_supreme",
            "Another-Dataset-on-used-Fiat-500-(1538-rows)",
            "archive2",
            "archive_r56_Maths",
            "archive_r56_Portuguese",
            "artificial-characters",
            "ASP-POTASSCO-classification",
            "auction_verification",
            "autoUniv-au4-2500",
            "autoUniv-au7-1100",
            "avocado_sales",
            "bank",
            "bank32nh",
            "bank8FM",
            "Bank_Customer_Churn_Dataset",
            "banknote_authentication",
            "baseball",
            "Basketball_c",
            "Bias_correction_r",
            "Bias_correction_r_2",
            "BLE_RSSI_dataset_for_Indoor_localization",
            "blogfeedback",
            "BNG(breast-w)",
            "BNG(cmc)",
            "BNG(echoMonths)",
            "BNG(lowbwt)",
            "BNG(mv)",
            "BNG(stock)",
            "BNG(tic-tac-toe)",
            "Brazilian_houses_reproduced",
            "California-Housing-Classification",
            "Cardiovascular-Disease-dataset",
            "car-evaluation",
            "CDC_Diabetes_Health_Indicators",
            "churn",
            "Click_prediction_small",
            "cmc",
            "combined_cycle_power_plant",
            "communities_and_crime",
            "company_bankruptcy_prediction",
            "compass",
            "compass_reg",
            "concrete_compressive_strength",
            "Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-10.0GHz(Urbinati)",
            "Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-10.5GHz(Urbinati)",
            "Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-11.0GHz(Urbinati)",
            "Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-9.0GHz(Urbinati)",
            "Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-9.5GHz(Urbinati)",
            "contraceptive_method_choice",
            "CookbookReviews",
            "CPMP-2015-regression",
            "CPMP-2015-runtime-regression",
            "CPS1988",
            "cpu_act",
            "cpu_small",
            "credit",
            "Credit_c",
            "credit_reg",
            "Customer_Personality_Analysis",
            "customer_satisfaction_in_airline",
            "dabetes_130-us_hospitals",
            "Data_Science_for_Good_Kiva_Crowdfunding",
            "Data_Science_Salaries",
            "dataset_sales",
            "debutanizer",
            "default_of_credit_card_clients",
            "delta_ailerons",
            "delta_elevators",
            "Diabetic_Retinopathy_Debrecen",
            "Diamonds",
            "dis",
            "dna",
            "drug_consumption",
            "dry_bean_dataset",
            "E-CommereShippingData",
            "eeg-eye-state",
            "electricity",
            "elevators",
            "Employee",
            "estimation_of_obesity_levels",
            "eye_movements",
            "eye_movements_bin",
            "Facebook_Comment_Volume",
            "FICO-HELOC-cleaned",
            "fifa",
            "Firm-Teacher_Clave-Direction_Classification",
            "first-order-theorem-proving",
            "Fitness_Club_c",
            "Food_Delivery_Time",
            "FOREX_audcad-day-High",
            "FOREX_audcad-hour-High",
            "FOREX_audchf-day-High",
            "FOREX_audjpy-day-High",
            "FOREX_audjpy-hour-High",
            "FOREX_audsgd-hour-High",
            "FOREX_audusd-hour-High",
            "FOREX_cadjpy-day-High",
            "FOREX_cadjpy-hour-High",
            "fried",
            "GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1",
            "GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001",
            "garments_worker_productivity",
            "gas-drift",
            "gas_turbine_CO_and_NOx_emission",
            "Gender_Gap_in_Spanish_WP",
            "GesturePhaseSegmentationProcessed",
            "gina_agnostic",
            "golf_play_dataset_extended",
            "Goodreads-Computer-Books",
            "healthcare_insurance_expenses",
            "Heart-Disease-Dataset-(Comprehensive)",
            "heloc",
            "hill-valley",
            "house_16H",
            "house_16H_reg",
            "house_8L",
            "houses",
            "house_sales_reduced",
            "housing_price_prediction",
            "HR_Analytics_Job_Change_of_Data_Scientists",
            "htru",
            "ibm-employee-performance",
            "IBM_HR_Analytics_Employee_Attrition_and_Performance",
            "IEEE80211aa-GATS",
            "Indian_pines",
            "INNHotelsGroup",
            "Insurance",
            "internet_firewall",
            "internet_usage",
            "Intersectional-Bias-Assessment",
            "in_vehicle_coupon_recommendation",
            "Is-this-a-good-customer",
            "JapaneseVowels",
            "jm1",
            "Job_Profitability",
            "jungle_chess_2pcs_raw_endgame_complete",
            "Kaggle_bike_sharing_demand_challange",
            "kc1",
            "KDD",
            "KDDCup09_upselling",
            "kdd_ipums_la_97-small",
            "kin8nm",
            "kropt",
            "kr-vs-k",
            "Laptop_Prices_Dataset",
            "Large-scale_Wave_Energy_Farm_Perth_100",
            "Large-scale_Wave_Energy_Farm_Perth_49",
            "Large-scale_Wave_Energy_Farm_Sydney_100",
            "Large-scale_Wave_Energy_Farm_Sydney_49",
            "law-school-admission-bianry",
            "led24",
            "led7",
            "letter",
            "Long",
            "madeline",
            "MagicTelescope",
            "mammography",
            "Marketing_Campaign",
            "maternal_health_risk",
            "mauna-loa-atmospheric-co2",
            "mfeat-factors",
            "mfeat-fourier",
            "mfeat-karhunen",
            "mfeat-morphological",
            "mfeat-pixel",
            "mfeat-zernike",
            "MiamiHousing2016",
            "MIC",
            "mice_protein_expression",
            "microaggregation2",
            "mobile_c36_oversampling",
            "Mobile_Phone_Market_in_Ghana",
            "Mobile_Price_Classification",
            "mozilla4",
            "mv",
            "NASA_PHM2008",
            "naticusdroid+android+permissions+dataset",
            "National_Health_and_Nutrition_Health_Survey",
            "national-longitudinal-survey-binary",
            "NHANES_age_prediction",
            "okcupid_stem",
            "one-hundred-plants-margin",
            "one-hundred-plants-shape",
            "one-hundred-plants-texture",
            "online_shoppers",
            "optdigits",
            "ozone_level",
            "ozone-level-8hr",
            "page-blocks",
            "Parkinson_Multiple_Sound_Recording",
            "Parkinsons_Telemonitoring",
            "pc1",
            "pc3",
            "pc4",
            "pendigits",
            "Performance-Prediction",
            "PhishingWebsites",
            "phoneme",
            "Physicochemical_r",
            "PieChart3",
            "Pima_Indians_Diabetes_Database",
            "PizzaCutter3",
            "pol",
            "pole",
            "pol_reg",
            "predict_students_dropout_and_academic_success",
            "puma32H",
            "puma8NH",
            "Pumpkin_Seeds",
            "qsar",
            "qsar_aquatic_toxicity",
            "QSAR_biodegradation",
            "qsar_fish_toxicity",
            "Rain_in_Australia",
            "rice_cammeo_and_osmancik",
            "ringnorm",
            "rl",
            "Satellite",
            "satellite_image",
            "satimage",
            "SDSS17",
            "segment",
            "seismic+bumps",
            "semeion",
            "sensory",
            "shill-bidding",
            "Shipping",
            "Shop_Customer_Data",
            "shrutime",
            "shuttle",
            "Smoking_and_Drinking_Dataset_with_body_signal",
            "socmob",
            "space_ga",
            "spambase",
            "splice",
            "sports_articles_for_objectivity_analysis",
            "statlog",
            "steel_industry_energy_consumption",
            "steel_plates_faults",
            "stock",
            "stock_fardamento02",
            "Student_Alcohol_Consumption",
            "Student_Performance_Portuguese",
            "sulfur",
            "Superconductivty",
            "svmguide3",
            "sylvine",
            "taiwanese_bankruptcy_prediction",
            "telco-customer-churn",
            "Telecom_Churn_Dataset",
            "texture",
            "thyroid",
            "thyroid-ann",
            "thyroid-dis",
            "topo_2_1",
            "treasury",
            "turiye_student_evaluation",
            "twonorm",
            "UJIndoorLoc",
            "UJI_Pen_Characters",
            "vehicle",
            "volkert",
            "volume",
            "VulNoneVul",
            "walking-activity",
            "wall-robot-navigation",
            "water_quality",
            "Water_Quality_and_Potability",
            "Waterstress",
            "waveform-5000",
            "waveform_database_generator",
            "waveform_database_generator_version_1",
            "weather_izmir",
            "website_phishing",
            "Wilt",
            "wind",
            "wine",
            "wine+quality",
            "Wine_Quality_red",
            "wine-quality-red",
            "Wine_Quality_white",
            "wine-quality-white",
            "yeast",
        ]

    def __init__(self, name, task_id=None):
        super().__init__(name, task_id)

    def prepare_data(self, download_dir):
        sub_dir = Path(download_dir) / "talent"
        if not sub_dir.exists():
            zip_path = os.path.join(download_dir, "talent.zip")
            gdown.download(id="1-dzY-BhMzcqjCM8vMTkVwa0hOYQ1598T", output=zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(sub_dir)
            os.remove(zip_path)

        # Data is split into numeric 'N' and categorical 'C' files
        N_splits = []
        C_splits = []
        y_splits = []
        for split in ("train", "val", "test"):
            npy = sub_dir / "data" / self.name / f"N_{split}.npy"
            if npy.exists():
                N_splits.append(np.load(npy, allow_pickle=True).astype(np.float32))
            npy = sub_dir / "data" / self.name / f"C_{split}.npy"
            if npy.exists():
                C_splits.append(np.load(npy, allow_pickle=True))
            y_splits.append(
                np.load(sub_dir / "data" / self.name / f"y_{split}.npy", allow_pickle=True)
            )
        if C_splits:
            C = OrdinalEncoder().fit_transform(np.concatenate(C_splits, axis=0))
            if N_splits:
                N = np.concatenate(N_splits, axis=0)
                self.X = np.concatenate((N, C), axis=1)
                self.metadata["categorical_feature_inds"] = [
                    i + N.shape[1] for i in range(C.shape[1])
                ]
            else:
                self.X = C
                self.metadata["categorical_feature_inds"] = list(range(C.shape[1]))
        else:
            self.X = np.concatenate(N_splits, axis=0)
        self.y = np.concatenate(y_splits, axis=0)
        if self.y.dtype == "object":
            self.y = LabelEncoder().fit_transform(self.y)
            self.metadata["target_type"] = "classification"
        else:
            self.metadata["target_type"] = "regression"
        self.y = self.y.squeeze()

        train_len, val_len, test_len = len(y_splits[0]), len(y_splits[1]), len(y_splits[2])
        self._train_inds = range(train_len)
        self._val_inds = range(train_len, train_len + val_len)
        self._test_inds = range(train_len + val_len, train_len + val_len + test_len)

    def all_instances(self):
        return self.X, self.y

    def train_inds(self):
        return self._train_inds

    def val_inds(self):
        return self._val_inds

    def test_inds(self):
        return self._test_inds
