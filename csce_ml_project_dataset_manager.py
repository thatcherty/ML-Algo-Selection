# The code is for a university class project and has no warranties of any kind.
import ucimlrepo
import time
import numpy as np
import pandas as pd
import pathlib
from rich.progress import Progress
import pickle
from typing import Optional, Union
import pathlib



class DataSetManager():


    def __init__(self, save_dir:Optional[Union[pathlib.Path, str]]=None):
        default_save_dir = 'project_datasets'
        if save_dir is None:
            self.save_dir = pathlib.Path(default_save_dir)
        elif isinstance(save_dir, (pathlib.Path,str)):
            self.save_dir = save_dir
        print(f"Save dir is: {self.save_dir}")

        self.datasets = [
            ["Abalone"                                                                                , 1],     
            ["Adult"                                                                                  , 2],     
            ["Annealing"                                                                              , 3],     
            ["Audiology (Standardized)"                                                               , 8],     
            ["Auto MPG"                                                                               , 9],     
            ["Automobile"                                                                             , 10],    
            ["Balance Scale"                                                                          , 12],    
            ["Balloons"                                                                               , 13],    
            ["Breast Cancer"                                                                          , 14],    
            ["Breast Cancer Wisconsin (Original)"                                                     , 15],    
            ["Breast Cancer Wisconsin (Prognostic)"                                                   , 16],    
            ["Breast Cancer Wisconsin (Diagnostic)"                                                   , 17],    
            ["Pittsburgh Bridges"                                                                     , 18],    
            ["Car Evaluation"                                                                         , 19],    
            ["Census Income"                                                                          , 20],    
            ["Chess (King-Rook vs. King-Pawn)"                                                        , 22],    
            ["Chess (King-Rook vs. King)"                                                             , 23],    
            ["Connect-4"                                                                              , 26],    
            ["Credit Approval"                                                                        , 27],    
            ["Japanese Credit Screening"                                                              , 28],    
            ["Computer Hardware"                                                                      , 29],    
            ["Contraceptive Method Choice"                                                            , 30],    
            ["Covertype"                                                                              , 31],    
            ["Cylinder Bands"                                                                         , 32],    
            ["Dermatology"                                                                            , 33],    
            ["Echocardiogram"                                                                         , 38],    
            ["Ecoli"                                                                                  , 39],    
            ["Flags"                                                                                  , 40],    
            ["Glass Identification"                                                                   , 42],    
            ["Haberman's Survival"                                                                    , 43],    
            ["Hayes-Roth"                                                                             , 44],    
            ["Heart Disease"                                                                          , 45],    
            ["Hepatitis"                                                                              , 46],    
            ["Horse Colic"                                                                            , 47],    
            ["Image Segmentation"                                                                     , 50],    
            ["Ionosphere"                                                                             , 52],    
            ["Iris"                                                                                   , 53],    
            ["ISOLET"                                                                                 , 54],    
            ["Lenses"                                                                                 , 58],    
            ["Letter Recognition"                                                                     , 59],    
            ["Liver Disorders"                                                                        , 60],    
            ["Lung Cancer"                                                                            , 62],    
            ["Lymphography"                                                                           , 63],    
            ["Molecular Biology (Splice-junction Gene Sequences)"                                     , 69],    
            ["MONK's Problems"                                                                        , 70],    
            ["Mushroom"                                                                               , 73],    
            ["Musk (Version 1)"                                                                       , 74],    
            ["Musk (Version 2)"                                                                       , 75],    
            ["Nursery"                                                                                , 76],    
            ["Page Blocks Classification"                                                             , 78],    
            ["Optical Recognition of Handwritten Digits"                                              , 80],    
            ["Pen-Based Recognition of Handwritten Digits"                                            , 81],    
            ["Post-Operative Patient"                                                                 , 82],    
            ["Primary Tumor"                                                                          , 83],    
            ["Servo"                                                                                  , 87],    
            ["Shuttle Landing Control"                                                                , 88],    
            ["Solar Flare"                                                                            , 89],    
            ["Soybean (Large)"                                                                        , 90],    
            ["Soybean (Small)"                                                                        , 91],    
            ["Challenger USA Space Shuttle O-Ring"                                                    , 92],    
            ["Spambase"                                                                               , 94],    
            ["SPECT Heart"                                                                            , 95],    
            ["SPECTF Heart"                                                                           , 96],    
            ["Tic-Tac-Toe Endgame"                                                                    , 101],   
            ["Congressional Voting Records"                                                           , 105],   
            ["Waveform Database Generator (Version 1)"                                                , 107],   
            ["Wine"                                                                                   , 109],   
            ["Yeast"                                                                                  , 110],   
            ["Zoo"                                                                                    , 111],   
            ["US Census Data (1990)"                                                                  , 116],   
            ["Census-Income (KDD)"                                                                    , 117],   
            ["El Nino"                                                                                , 122],   
            ["Statlog (Australian Credit Approval)"                                                   , 143],   
            ["Statlog (German Credit Data)"                                                           , 144],   
            ["Statlog (Heart)"                                                                        , 145],   
            ["Statlog (Landsat Satellite)"                                                            , 146],   
            ["Statlog (Image Segmentation)"                                                           , 147],   
            ["Statlog (Shuttle)"                                                                      , 148],   
            ["Statlog (Vehicle Silhouettes)"                                                          , 149],   
            ["Connectionist Bench (Sonar, Mines vs. Rocks)"                                           , 151],   
            ["Cloud"                                                                                  , 155],   
            ["Poker Hand"                                                                             , 158],   
            ["MAGIC Gamma Telescope"                                                                  , 159],   
            ["Mammographic Mass"                                                                      , 161],   
            ["Forest Fires"                                                                           , 162],   
            ["Concrete Compressive Strength"                                                          , 165],   
            ["Ozone Level Detection"                                                                  , 172],   
            ["Parkinsons"                                                                             , 174],   
            ["Blood Transfusion Service Center"                                                       , 176],   
            ["Communities and Crime"                                                                  , 183],   
            ["Acute Inflammations"                                                                    , 184],   
            ["Wine Quality"                                                                           , 186],   
            ["Parkinsons Telemonitoring"                                                              , 189],   
            ["Cardiotocography"                                                                       , 193],   
            ["Steel Plates Faults"                                                                    , 198],   
            ["Communities and Crime Unnormalized"                                                     , 211],   
            ["Vertebral Column"                                                                       , 212],   
            ["Bank Marketing"                                                                         , 222],   
            ["ILPD (Indian Liver Patient Dataset)"                                                    , 225],   
            ["Skin Segmentation"                                                                      , 229],   
            ["Individual Household Electric Power Consumption"                                        , 235],   
            ["Energy Efficiency"                                                                      , 242],   
            ["Fertility"                                                                              , 244],   
            ["ISTANBUL STOCK EXCHANGE"                                                                , 247],   
            ["User Knowledge Modeling"                                                                , 257],   
            ["EEG Eye State"                                                                          , 264],   
            ["Banknote Authentication"                                                                , 267],   
            ["Gas Sensor Array Drift at Different Concentrations"                                     , 270],   
            ["Bike Sharing"                                                                           , 275],   
            ["Thoracic Surgery Data"                                                                  , 277],   
            ["Airfoil Self-Noise"                                                                     , 291],   
            ["Wholesale customers"                                                                    , 292],   
            ["Combined Cycle Power Plant"                                                             , 294],   
            ["Diabetes 130-US Hospitals for Years 1999-2008"                                          , 296],   
            ["Tennis Major Tournament Match Statistics"                                               , 300],   
            ["Dow Jones Index"                                                                        , 312],   
            ["Student Performance"                                                                    , 320],   
            ["Phishing Websites"                                                                      , 327],   
            ["Diabetic Retinopathy Debrecen"                                                          , 329],   
            ["Online News Popularity"                                                                 , 332],   
            ["Chronic Kidney Disease"                                                                 , 336],   
            ["Mice Protein Expression"                                                                , 342],   
            ["Default of Credit Card Clients"                                                         , 350],   
            ["Online Retail"                                                                          , 352],   
            ["Occupancy Detection"                                                                    , 357],   
            ["Air Quality"                                                                            , 360],   
            ["Polish Companies Bankruptcy"                                                            , 365],   
            ["Dota2 Games Results"                                                                    , 367],   
            ["Facebook Metrics"                                                                       , 368],   
            ["HTRU2"                                                                                  , 372],   
            ["Drug Consumption (Quantified)"                                                          , 373],   
            ["Appliances Energy Prediction"                                                           , 374],   
            ["Website Phishing"                                                                       , 379],   
            ["YouTube Spam Collection"                                                                , 380],   
            ["Beijing PM2.5"                                                                          , 381],   
            ["Cervical Cancer (Risk Factors)"                                                         , 383],   
            ["Stock Portfolio Performance"                                                            , 390],   
            ["Sales Transactions Weekly"                                                              , 396],   
            ["Daily Demand Forecasting Orders"                                                        , 409],   
            ["Autistic Spectrum Disorder Screening Data for Children"                                 , 419],   
            ["Autism Screening Adult"                                                                 , 426],   
            ["Absenteeism at work"                                                                    , 445],   
            ["Breast Cancer Coimbra"                                                                  , 451],   
            ["Drug Reviews (Druglib.com)"                                                             , 461],   
            ["Drug Reviews (Drugs.com)"                                                               , 462],   
            ["Superconductivty Data"                                                                  , 464],   
            ["Student Academics Performance"                                                          , 467],   
            ["Online Shoppers Purchasing Intention Dataset"                                           , 468],   
            ["Electrical Grid Stability Simulated Data"                                               , 471],   
            ["Real Estate Valuation"                                                                  , 477],   
            ["Travel Reviews"                                                                         , 484],   
            ["Travel Review Ratings"                                                                  , 485],   
            ["Facebook Live Sellers in Thailand"                                                      , 488],   
            ["Metro Interstate Traffic Volume"                                                        , 492],   
            ["Hepatitis C Virus (HCV) for Egyptian patients"                                          , 503],   
            ["Heart Failure Clinical Records"                                                         , 519],   
            ["Early Stage Diabetes Risk Prediction"                                                   , 529],   
            ["Pedestrians in Traffic"                                                                 , 536],   
            ["Cervical Cancer Behavior Risk"                                                          , 537],   
            ["Estimation of Obesity Levels Based On Eating Habits and Physical Condition"             , 544],   
            ["Rice (Cammeo and Osmancik)"                                                             , 545],   
            ["Algerian Forest Fires"                                                                  , 547],   
            ["Gas Turbine CO and NOx Emission Data Set"                                               , 551],   
            ["Apartment for Rent Classified"                                                          , 555],   
            ["Seoul Bike Sharing Demand"                                                              , 560],   
            ["Iranian Churn"                                                                          , 563],   
            ["Bone marrow transplant: children"                                                       , 565],   
            ["COVID-19 Surveillance"                                                                  , 567],   
            ["HCV data"                                                                               , 571],   
            ["Taiwanese Bankruptcy Prediction"                                                        , 572],   
            ["Myocardial infarction complications"                                                    , 579],   
            ["Student Performance on an Entrance Examination"                                         , 582],   
            ["Gender by Name"                                                                         , 591],   
            ["Productivity Prediction of Garment Employees"                                           , 597],   
            ["AI4I 2020 Predictive Maintenance Dataset"                                               , 601],   
            ["Dry Bean"                                                                               , 602],   
            ["In-Vehicle Coupon Recommendation"                                                       , 603],   
            ["Predict Students' Dropout and Academic Success"                                         , 697],   
            ["Auction Verification"                                                                   , 713],   
            ["NATICUSdroid (Android Permissions)"                                                     , 722],   
            ["Toxicity"                                                                               , 728],   
            ["DARWIN"                                                                                 , 732],   
            ["Accelerometer Gyro Mobile Phone"                                                        , 755],   
            ["Glioma Grading Clinical and Mutation Features"                                          , 759],   
            ["Multivariate Gait Data"                                                                 , 760],   
            ["Land Mines"                                                                             , 763],   
            ["Single Elder Home Monitoring: Gas and Position"                                         , 799],   
            ["Sepsis Survival Minimal Clinical Records"                                               , 827],   
            ["Secondary Mushroom"                                                                     , 848],   
            ["Power Consumption of Tetouan City"                                                      , 849],   
            ["Raisin"                                                                                 , 850],   
            ["Steel Industry Energy Consumption"                                                      , 851],   
            ["Higher Education Students Performance Evaluation"                                       , 856],   
            ["Risk Factor Prediction of Chronic Kidney Disease"                                       , 857],   
            ["Maternal Health Risk"                                                                   , 863],   
            ["Room Occupancy Estimation"                                                              , 864],   
            ["Cirrhosis Patient Survival Prediction"                                                  , 878],   
            ["SUPPORT2"                                                                               , 880],   
            ["National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset"   , 887],   
            ["AIDS Clinical Trials Group Study 175"                                                   , 890],   
            ["CDC Diabetes Health Indicators"                                                         , 891],   
            ["Recipe Reviews and User Feedback"                                                       , 911],   
            ["Forty Soybean Cultivars from Subsequent Harvests"                                       , 913],   
            ["Differentiated Thyroid Cancer Recurrence"                                               , 915],   
            ["Infrared Thermography Temperature"                                                      , 925],   
            ["National Poll on Healthy Aging (NPHA)"                                                  , 936],   
            ["Regensburg Pediatric Appendicitis"                                                      , 938],   
            ["RT-IoT2022"                                                                             , 942],   
            ["PhiUSIIL Phishing URL (Website)"                                                        , 967],
        ]
        self.id_list = np.array([x[1] for x in self.datasets])
        self.dataset_dict = {}
        self.filter_dict = {}

        # Load our pregenerated long dataframe if it exists.
        df_long_all_load_path = self.save_dir / "df_long_all.parquet"
        if df_long_all_load_path.exists():
            self.df_long_all = pd.read_parquet(df_long_all_load_path)
        else:
            print(f"df_long not loaded: {df_long_all_load_path}")
            self.df_long_all = None

        # Load our pregenerated filtered dataframe if it exists.
        df_filtered_load_path = self.save_dir / "df_filtered.parquet"
        if df_filtered_load_path.exists():
            self.df_filtered = pd.read_parquet(df_filtered_load_path)
        else:
            print(f"df_filtered not loaded: {df_filtered_load_path}")
            self.df_filtered = None


    def save(self, data, id, force_overwrite=False):
        """
        Save a dataset locally.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{id}.pkl"
        save_path = self.save_dir / filename
        if not save_path.exists() or force_overwrite:
            with open(save_path, "wb") as f:
                pickle.dump(data, f)
        else:
            print(f"Cannot save dataset: {save_path}")


    def load(self, id_list=None):
        """
        Load a locally saved dataset.
        """
        if id_list is None:
            id_list = self.id_list
        loaded_id_list = []
        failed_id_list = []
        with Progress() as progress:
            task = progress.add_task("Loading...", total=len(id_list))
            progress.update(task, advance=len(loaded_id_list)+len(failed_id_list), description=f"")
            for id in id_list:
                progress.update(task, advance=0, description=f"Loading [{id:>4}]")
                load_path = f"{str(self.save_dir)}/{id}.pkl"
                try:
                    with open(load_path, "rb") as f:
                        dataset = pickle.load(f)
                    self.dataset_dict[id] = dataset
                    loaded_id_list.append(id)
                except Exception as e:
                    failed_id_list.append(id)
                    print(f"Exception loading: [{f"{id}"}] {e}")
                finally:
                    progress.update(task, advance=1)
                    if id==id_list[-1]:
                        time.sleep(.25)
                        progress.update(task, description=f"Done")
            print(f"Loaded datasets: {len(loaded_id_list)}")
            if len(failed_id_list) > 0:
                print(f"Failed to load: {len(failed_id_list)}: {[f"{x}" for x in failed_id_list]}")


    def get_dataset_list(self, id_list:list=list()):
        """
        Get the datasets in a list.
        """
        if len(id_list) == 0:
            return [self.get_dataset(id) for id in self.dataset_dict.keys()]
        else:
            return [self.get_dataset(id) for id in id_list]


    def fetch(self, id_list=None, save=False, pretend=False):
        """
        Download datasets.
        """
        if id_list is None:
            datasets = self.datasets
        else:
            datasets = [x for x in self.datasets if x[1] in id_list]    
        items = len(datasets)
        self.fetched_ids = []
        self.failed_fetch_ids = []
        num=0
        with Progress() as progress:
            task = progress.add_task("Fetching...", total=items)
            progress.update(task, advance=len(self.fetched_ids)+len(self.failed_fetch_ids))
            for x in datasets:
                try:
                    num = x[1]
                    progress.update(task, advance=0, description=f"Fetching [{num:>4}]")
                    if num not in set(set(self.fetched_ids)|set(self.failed_fetch_ids)):
                        if not pretend:
                            dataset = ucimlrepo.fetch_ucirepo(id=num)
                            if save:
                                self.save(id=num, data=dataset)
                            
                            self.dataset_dict[num] = dataset
                        self.fetched_ids.append(num)

                        time.sleep(.2)
                except OSError as e:
                    print(f"Exception: [{x[0]} id: {x[1]}]  {e}")
                except Exception as e:
                    if isinstance(num, int):
                        if num not in self.fetched_ids:
                            self.failed_fetch_ids.append(num)
                    print(f"Exception fetching: [{x[0]} id: {x[1]}]  {e}")
                finally:
                    progress.update(task, advance=1)
                    if x==datasets[-1]:
                        time.sleep(.25)
                        progress.update(task, description=f"Done")


    def unload_all(self):
        """
        Reinitialize the dataset dict.
        """
        self.dataset_dict = {}


    def generate_metadata_df(self, verbose=0):
        """
        Build several dataframes from the dataset metadata.
        Expects most of the datasets to be loaded.
        """
        metadata_list = []
        for k,x in self.dataset_dict.items():
            if verbose >= 1: 
                print(x['metadata'])
            metadata_list.append(x['metadata'])
        df_long = pd.json_normalize(metadata_list)
        short_column_list = [
            "uci_id"                                    , # 207 non-null    int64  
            "name"                                      , # 207 non-null    str    
            # "repository_url"                            , # 207 non-null    str    
            # "data_url"                                  , # 207 non-null    str    
            # "abstract"                                  , # 207 non-null    str    
            "area"                                      , # 207 non-null    str    
            "tasks"                                     , # 207 non-null    object 
            "characteristics"                           , # 207 non-null    object 
            "num_instances"                             , # 207 non-null    int64  
            "num_features"                              , # 207 non-null    int64  
            "feature_types"                             , # 207 non-null    object 
            "demographics"                              , # 207 non-null    object 
            "target_col"                                , # 181 non-null    object 
            "index_col"                                 , # 51 non-null     object 
            "has_missing_values"                        , # 207 non-null    str    
            "missing_values_symbol"                     , # 49 non-null     str    
            # "year_of_dataset_creation"                  , # 203 non-null    float64
            # "last_updated"                              , # 205 non-null    str    
            # "dataset_doi"                               , # 207 non-null    str    
            # "creators"                                  , # 207 non-null    object 
            # "intro_paper"                               , # 0 non-null      float64
            "additional_info.summary"                   , # 187 non-null    str    
            # "additional_info.purpose"                   , # 20 non-null     str    
            # "additional_info.funded_by"                 , # 14 non-null     str    
            # "additional_info.instances_represent"       , # 24 non-null     str    
            # "additional_info.recommended_data_splits"   , # 12 non-null     str    
            "additional_info.sensitive_data"            , # 13 non-null     str    
            # "additional_info.preprocessing_description" , # 15 non-null     str    
            # "additional_info.variable_info"             , # 180 non-null    str    
            # "additional_info.citation"                  , # 31 non-null     str    
            # "intro_paper.ID"                            , # 109 non-null    float64
            # "intro_paper.type"                          , # 109 non-null    str    
            # "intro_paper.title"                         , # 109 non-null    str    
            # "intro_paper.authors"                       , # 109 non-null    str    
            # "intro_paper.venue"                         , # 109 non-null    str    
            # "intro_paper.year"                          , # 109 non-null    float64
            # "intro_paper.journal"                       , # 9 non-null      str    
            # "intro_paper.DOI"                           , # 38 non-null     str    
            # "intro_paper.URL"                           , # 106 non-null    str    
            # "intro_paper.sha"                           , # 2 non-null      str    
            # "intro_paper.corpus"                        , # 3 non-null      str    
            # "intro_paper.arxiv"                         , # 0 non-null      float64
            # "intro_paper.mag"                           , # 0 non-null      float64
            # "intro_paper.acl"                           , # 0 non-null      float64
            # "intro_paper.pmid"                          , # 5 non-null      str    
            # "intro_paper.pmcid"                         , # 0 non-null      float64
            # "external_url"                              , # 5 non-null      str    
            "tasks__Causa"                              , # 207 non-null    uint8  
            "tasks__Causal-Discovery"                   , # 207 non-null    uint8  
            "tasks__Classification"                     , # 207 non-null    uint8  
            "tasks__Clustering"                         , # 207 non-null    uint8  
            "tasks__Other"                              , # 207 non-null    uint8  
            "tasks__Regression"                         , # 207 non-null    uint8  
            "characteristics__Data-Generator"           , # 207 non-null    uint8  
            "characteristics__Domain-Theory"            , # 207 non-null    uint8  
            "characteristics__Image"                    , # 207 non-null    uint8  
            "characteristics__Multivariate"             , # 207 non-null    uint8  
            "characteristics__Other"                    , # 207 non-null    uint8  
            "characteristics__Sequential"               , # 207 non-null    uint8  
            "characteristics__Spatial"                  , # 207 non-null    uint8  
            "characteristics__Tabular"                  , # 207 non-null    uint8  
            "characteristics__Text"                     , # 207 non-null    uint8  
            "characteristics__Time-Series"              , # 207 non-null    uint8  
            "characteristics__Univariate"               , # 207 non-null    uint8  
            "feature_types__Categorical"                , # 207 non-null    uint8  
            "feature_types__Integer"                    , # 207 non-null    uint8  
            "feature_types__Real"                       , # 207 non-null    uint8  
        ]
        if not 'tasks__Classification' in df_long.columns:
            for col in ['tasks','characteristics','feature_types',]:
                dummies = (
                    df_long[col]
                    .apply(lambda x: x if isinstance(x, list) else [])
                    .str.join("|")
                    .str.get_dummies(sep="|")
                    .astype("uint8")
                    .add_prefix(f"{col}__")
                )
                df_long = df_long.join(dummies)
                self.df_long = df_long

            df_short = df_long[short_column_list]
            sensitive_info_mask = df_short["additional_info.sensitive_data"].isin([        
                pd.NA,
                # 'Yes. The data contains information about the age and gender of the patients.',
                # 'Yes. The dataset contains information about the age, gender, and race of the patients.',
                # 'There is information about race, age, and gender of the patient.',
                'No',
                'No.',
                # 'Yes. It contains information about the gender and age of the patient.',
                # 'Gender, Age',
                # 'Yes. There is information about race, gender, income, and education level.',
                # '- Ethnicity (race)\n- Gender',
                # '- Gender\n- Income\n- Education level',
                'No data is confidential',
                # 'Yes. There is information about race/ethnicity, gender, age.',    
            ])
            filtered_df = df_short[sensitive_info_mask]
            self.filtered_df = filtered_df
            df = filtered_df
            subset_df = df[
                    (df['tasks__Classification']==1) &
                    (df['characteristics__Time-Series']==0) &
                    (df['has_missing_values']=='no') 
                ]
            self.subset_df = subset_df
            print(f"Filtered DataFrame:        self.filtered_df")
            print(f"Subset for Classification: self.df_subset")


    def get_dataset(self, dataset_id):
        """
        Returns the dataset with dataset_id.
        """
        return self.dataset_dict[dataset_id]
    
    
    def save_df(self):
        """
        Saves several metadata dataframes locally.
        When this object is initialized, these dataframes will be loaded 
        if they exist so that they do not have to be recreated on each instantiation.
        We assume the underlying data does not change.
        """
        filepath = self.save_dir / f"df_long_all.parquet"
        self.df_long.to_parquet(filepath)
        filepath = self.save_dir / f"df_filtered.parquet"
        self.filtered_df.to_parquet(filepath)

           
