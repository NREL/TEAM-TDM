import local_conf

os.environ['PATH'] = local_conf.MINGW_PATH + ';' + os.environ['PATH']

from xgboost import XGBClassifier
from xgboost import XGBRegressor