import pandas as pd
import joblib # to use the saved the preprocessed training data after encoding, power transformation, and resampling

from evidently import Report
from evidently.presets import DataDriftPreset
from config.paths_config import * 
from src.logger import get_logger
from src.custom_exception import CustomException
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency
from src.artifact_store import load_joblib_artifact, save_joblib_artifact
from config.mlflow_config import configure_mlflow

configure_mlflow()

logger = get_logger(__name__)

class DataDriftMonitor:

    def __init__(self, p_val=0.05):
        self.reference_df = None
        self.current_df = None
        self.p_val = p_val
        self.detector = None
        self.feature_names = None

    def load_reference_data(self, path):
        try:
            self.reference_df = load_joblib_artifact(path) # X_train_res
            logger.info(f'Successfully loaded the reference data.. with {self.reference_df.shape} rows and columns')
        except Exception as e:
            raise CustomException(str(e),e)
        
    def load_current_data(self, path):
        try:
            self.current_df = pd.read_csv(path)
            logger.info(f'Successfully loaded the current data.. with {self.current_df.shape} rows and columns')
        except Exception as e:
            raise CustomException(str(e),e)
        
    def preprocess(self):
        try:

            ref = self.reference_df.copy()
            cur = self.current_df.copy()

            # Remove target
            # Reference data doesnt have target as it's processed final data

            if "greenwashing_flag" in cur.columns:
                cur = cur.drop(columns=["greenwashing_flag"])

            self.feature_names = ref.columns.tolist()

            # APPLY SAME PREPROCESSING STEP TO CURRENT DATA

            # Create flag -- This marks rows where the year-on-year percentage change is missing (which happens for the first year). 
            cur['is_first_year'] = cur['yoy_scope1_change_pct'].isnull().astype(int)

            # Impute nulls with 0 (or sentinel like -999)
            cur['yoy_scope1_change_pct'] = cur['yoy_scope1_change_pct'].fillna(-999)

            cur.drop(columns=['company','ticker'], inplace=True)

            cur['scope1_intensity'] = cur['scope1_emissions_mt_co2e'] / cur['revenue_usd_bn']
            cur['scope2_intensity'] = cur['scope2_emissions_mt_co2e'] / cur['revenue_usd_bn']
            cur['scope3_intensity'] = cur['scope3_emissions_mt_co2e'] / cur['revenue_usd_bn']

            cur['total_emissions'] = cur['scope1_emissions_mt_co2e'] + cur['scope2_emissions_mt_co2e'] + cur['scope3_emissions_mt_co2e']
                
            # ESG vs Emissions Mismatch as A company claiming high ESG but having poor carbon performance may look suspicious.

            cur['esg_emission_gap'] = (
                    cur['esg_score_0_100'] /
                    cur['carbon_intensity_tco2e_per_musd']
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
                cur[col] = cur[col].map({'Yes': 1, 'No': 0})

            cur['commitment_score'] = cur[binary_cols].sum(axis=1)

            cdp_map = {
                    'D': 1,
                    'C': 2,
                    'B-': 3,
                    'B': 4,
                    'A-': 5,
                    'A': 6
                }

            cur['cdp_climate_score'] = cur['cdp_climate_score'].map(cdp_map)

            # loading encoded and transformed weights
            encoder = load_joblib_artifact(ENCODED_PATH)
            power_transformer = load_joblib_artifact(POWER_TRANSFORMER_PATH)

            # Use exactly the columns the saved transformer was fitted on.
            skewed_features = list(power_transformer.feature_names_in_)
            cur[skewed_features] = power_transformer.transform(cur[skewed_features])

            # encoding
            categorical_cols = ['country', 'sector']

            encoder_cur = encoder.transform(cur[categorical_cols])

            encoded_cur = pd.DataFrame(encoder_cur,
                                    columns=encoder.get_feature_names_out(categorical_cols),
                                    index=cur.index)
            
            cur = pd.concat([cur.drop(columns=categorical_cols), encoded_cur], axis=1)

            missing_cols = [col for col in self.feature_names if col not in cur.columns]
            extra_cols = [col for col in cur.columns if col not in self.feature_names]

            if missing_cols:
                raise ValueError(f"Current data is missing required columns: {missing_cols}")

            if extra_cols:
                logger.warning(f"Dropping unexpected current-data columns: {extra_cols}")

            cur = cur.reindex(columns=self.feature_names) # Align column order with the training dataset or reference dataset

            os.makedirs(os.path.dirname(CURRENT_PROCESSED_DATA),
                        exist_ok=True)
            save_joblib_artifact(cur, CURRENT_PROCESSED_DATA)

            logger.info("Data preprocessing of new data done")
            return ref.values, cur.values 

        except Exception as e:
            raise CustomException(str(e), e)


    def detect_get_drift_summary(self):
        try:
            X_ref, X_cur = self.preprocess()
            
            # X_ref and X_cur are numpy arrays and evidently does not accept raw NumPy arrays

            X_ref_df = pd.DataFrame(X_ref, columns=self.feature_names)
            X_cur_df = pd.DataFrame(X_cur, columns=self.feature_names)

            report = Report(metrics=[DataDriftPreset()])

            # Drop columns with all values identical because Evidently computes correlations, and a column with zero standard deviation causes error

            # Compute constant columns in either dataset and remove them from both
            constant_cols = []

            for col in X_ref_df.columns:
                if (
                    X_ref_df[col].nunique() <= 1
                    or X_cur_df[col].nunique() <= 1
                ):
                    constant_cols.append(col)

            print("Removing constant columns:", constant_cols)

            X_ref_df = X_ref_df.drop(columns=constant_cols)
            X_cur_df = X_cur_df.drop(columns=constant_cols)
                        
            snapshot = report.run(
                reference_data=X_ref_df, 
                current_data=X_cur_df)
           
            snapshot.save_html("drift_report.html")

            logger.info('Drift detection done....')
  

            drifted_features = []

            numerical_cols = ['year', 'revenue_usd_bn', 'scope1_emissions_mt_co2e', 'scope2_emissions_mt_co2e',
                          'scope3_emissions_mt_co2e', 'total_s1_s2_mt_co2e', 'yoy_scope1_change_pct',
                         'carbon_intensity_tco2e_per_musd', 'esg_score_0_100', 'scope1_intensity', 
                         'scope2_intensity', 'scope3_intensity', 'total_emissions', 'esg_emission_gap', 
                         'commitment_score']
            
            numerical_cols = [
                col for col in numerical_cols
                if col not in constant_cols
            ]

            for col in numerical_cols:
                _,p = ks_2samp(X_ref_df[col], X_cur_df[col])

                if p < self.p_val:
                    drifted_features.append(col)
            
            # categorical cols
            categorical_col = [col for col in self.feature_names if col not in numerical_cols and col not in constant_cols]

            for col in categorical_col:
                contingency = pd.crosstab(X_ref_df[col], X_cur_df[col])

                _,p,_,_ = chi2_contingency(contingency)

                if p < self.p_val:
                    drifted_features.append(col)
            
            drift_score = len(drifted_features) / len(X_ref_df.columns) # 0.2 means 20% of features show drift.

            logger.info("Drift summary generated....")

            return {
                "drift_score": round(drift_score,4),
                "drift_detected": drift_score > 0,
                "drifted_features" : drifted_features
            }

        except Exception as e:
            raise CustomException(str(e), e)


    def run(self):
        self.load_reference_data(X_TRAIN_PATH)
        self.load_current_data(CURRENT_DATA_PATH)
        self.preprocess()
        self.detect_get_drift_summary()

if __name__ =='__main__':
    monitor = DataDriftMonitor()
    monitor.run()
    report = monitor.detect_get_drift_summary()
    print(report["drift_score"])
    print(report["drift_detected"])
    print(report["drifted_features"])

# For numerical the Kolmogorov–Smirnov (KS) test is used. 
# For categorical features, the Chi‑square test is more appropriate
