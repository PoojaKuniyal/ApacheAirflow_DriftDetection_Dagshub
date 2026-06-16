import psycopg2 # database driver used to connect to postgres sql, SQLAlchemy  uses it for connection
import sqlalchemy # use to work with databases
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import sys
import joblib
from config.database_config import *
from config.paths_config import *
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, db_params):
        self.db_params = db_params # environment variables (DB_CONFIG) is used for local testing and Airflow Connections when running in Airflow
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None 
        self.y_test = None 
        self.X_train_res = None
        self.y_train_res = None

    def connect_to_db_read_data(self):
        try:
            # Build connection string from db_params
            engine = sqlalchemy.create_engine(
                f"postgresql+psycopg2://{self.db_params['user']}:{self.db_params['password']}@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['dbname']}"
            )
            self.df = pd.read_sql('SELECT * FROM public."ESG"', engine)

            logger.info("Database Connection Established and data read successfully...")
            return self.df 

        except Exception as e:
            logger.error(f"Error while establishing connection {e}")
            raise CustomException(str(e), sys)

    def process_feature_engineer_data(self):
        try:
            self.df['scope1_intensity'] = self.df['scope1_emissions_mt_co2e'] / self.df['revenue_usd_bn']
            self.df['scope2_intensity'] = self.df['scope2_emissions_mt_co2e'] / self.df['revenue_usd_bn']
            self.df['scope3_intensity'] = self.df['scope3_emissions_mt_co2e'] / self.df['revenue_usd_bn']

            self.df['total_emissions'] = self.df['scope1_emissions_mt_co2e'] + self.df['scope2_emissions_mt_co2e'] + self.df['scope3_emissions_mt_co2e']
            
            # ESG vs Emissions Mismatch as A company claiming high ESG but having poor carbon performance may look suspicious.

            self.df['esg_emission_gap'] = (
                self.df['esg_score_0_100'] /
                self.df['carbon_intensity_tco2e_per_musd']
            )

            # # Climate Commitment Score
            # Binary columns
            binary_cols = [
                'net_zero_target_set',
                'sbti_committed',
                'emissions_disclosed',
                'third_party_verified'
            ]

            for col in binary_cols:
                self.df[col] = self.df[col].map({'Yes': 1, 'No': 0})

            self.df['commitment_score'] = self.df[binary_cols].sum(axis=1)

            cdp_map = {
                'D': 1,
                'C': 2,
                'B-': 3,
                'B': 4,
                'A-': 5,
                'A': 6
            }

            self.df['cdp_climate_score'] = self.df['cdp_climate_score'].map(cdp_map)

            logger.info("Categorical mapping & Feature creation done...")
        
        except Exception as e:
            logger.error(f"Error while feature engineering {e}")
            raise CustomException(str(e), sys)
    
    def split_data(self):
        try:
            X = self.df.drop(columns=['greenwashing_flag'])
            y = self.df['greenwashing_flag']

            # stratify=y → preserves class proportions during splitting, SMOTE → balances the training set after splitting.
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=67, stratify=y
            )

            logger.info('Splitting done...')
        
        except Exception as e:
            logger.error(f"Error while splitting data {e}")
            raise CustomException(str(e), sys)
        
    def transform_encode(self):
        try:
            
            # Select numeric columns to perform transformation
            numeric_df = self.X_train.select_dtypes(exclude='object')

            # Columns to exclude (categorical cols created after mapping)
            exclude = [
                'year','greenwashing_flag','cdp_climate_score',
                'net_zero_target_set','sbti_committed','emissions_disclosed',
                'third_party_verified','commitment_score'
            ]

            # Keep only numeric columns not in exclude list
            numeric_cols = [col for col in numeric_df.columns if col not in exclude]

            # Check skewness
            skewness = self.X_train[numeric_cols].apply(lambda x: skew(x.dropna()))

            # Identify highly skewed features
            skewed_features = skewness[skewness.abs() > 1].index

            # Apply Yeo-Johnson transformation
            pt = PowerTransformer(method='yeo-johnson')
            self.X_train[skewed_features] = pt.fit_transform(self.X_train[skewed_features])
            self.X_test[skewed_features] = pt.transform(self.X_test[skewed_features])


            # Define categorical columns
            categorical_cols = ['country', 'sector']

            # Initialize encoder
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

            # Fit on training set, transform both train and test
            encoded_train = encoder.fit_transform(self.X_train[categorical_cols])
            encoded_test = encoder.transform(self.X_test[categorical_cols])

            # Convert to DataFrames with proper column names
            encoded_df_train = pd.DataFrame(encoded_train, 
                                            columns=encoder.get_feature_names_out(categorical_cols),
                                            index=self.X_train.index)

            encoded_df_test = pd.DataFrame(encoded_test, 
                                        columns=encoder.get_feature_names_out(categorical_cols),
                                        index=self.X_test.index)

            # Drop original categorical columns and concatenate encoded ones
            self.X_train = pd.concat([self.X_train.drop(columns=categorical_cols), encoded_df_train], axis=1)
            self.X_test = pd.concat([self.X_test.drop(columns=categorical_cols), encoded_df_test], axis=1)

            # For future prediciton save
            os.makedirs(PROCESSED, exist_ok=True)

            os.makedirs(
                os.path.dirname(POWER_TRANSFORMER_PATH),
                exist_ok=True
            )
            joblib.dump(pt, POWER_TRANSFORMER_PATH)

            os.makedirs(
                os.path.dirname(ENCODED_PATH),
                exist_ok=True
            )
            joblib.dump(encoder, ENCODED_PATH)

            logger.info('Successfully imputed, transformed and encoded..')
        
        except Exception as e:
            logger.error(f"Error while imputing, transforming and encoding {e}")
            raise CustomException(str(e), sys)
        
    def apply_smote(self):
        try:
            # # Apply SMOTE only to training data
            smote = SMOTE(random_state=67)
            self.X_train_res, self.y_train_res = smote.fit_resample(self.X_train,self.y_train)
            
            joblib.dump(self.X_train_res, X_TRAIN_PATH)
            joblib.dump(self.X_test, X_TEST_PATH)

            joblib.dump(self.y_train_res, Y_TRAIN_PATH)
            joblib.dump(self.y_test, Y_TEST_PATH)

        except Exception as e:
            raise CustomException(str(e),sys) 
        
        
    def run(self):
        try:
            logger.info("Starting data processing pipeline..")
            self.connect_to_db_read_data() 
            self.process_feature_engineer_data()
            self.split_data()
            self.transform_encode()
            self.apply_smote()
            
            logger.info("End of data processing pipeline")
            return (
                    self.X_train_res,
                    self.X_test,
                    self.y_train_res,
                    self.y_test
                )
        except Exception as e:
            logger.error("Error while data processing..")
            raise CustomException(str(e),sys)

if __name__ =="__main__":
    data_processor = DataProcessing(DB_CONFIG)
    data_processor.run()

