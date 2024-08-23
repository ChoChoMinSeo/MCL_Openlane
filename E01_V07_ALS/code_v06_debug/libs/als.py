import os

import numpy as np
import pandas as pd
import findspark
findspark.init('')
print(findspark.find())
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS
# from pyspark.ml.recommendation import ALS
import sys
import os
from scipy.interpolate import interp1d
import pandas as pd
# from libs.extrapolation_modes import do_extrapolation

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


class ALS_pyspark(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.als_param = dict()
        self.als_param['rank'] = 3
        self.als_param['iterations'] = 20
        # self.als_param['lambda'] = 1e-12
        self.prepare_spark()

    def prepare_spark(self):
        conf = SparkConf().setAppName('Recommendations').setMaster('local')
        conf.set("spark.driver.memory", "10g")
        conf.set("spark.executor.memory", "10g")
        conf.set("spark.driver.cores", "4")

        # conf.setSystemProperty('spark.executor.memory', '2g')
        self.sc = SparkContext(conf=conf)  # instantiating spark context
        # self.spark = SparkSession.builder.master('local[2]').appName('Recommendations').config("spark.executer.memory", MAX_MEMORY).config("spark.driver.memory", MAX_MEMORY).\
        #     config('spark.driver.maxResultSize','1m').config("spark.executor.pyspark.memory", '1m').config("spark.cores.max", '2').config("spark.shuffle.push.merge.finalizeThreads", '2').getOrCreate()
        
        self.spark = SparkSession(self.sc)
        # check if spark context is defined
        print(self.spark.sparkContext._conf.getAll())
        print(self.sc.version)

    def convert_data_to_csr_matrix(self, data_x, trainable_idx):
        rows, cols = data_x.nonzero()
        vals_x = data_x[rows, cols]
        self.available_y = np.unique(rows)
        self.not_avail_y = np.array(range(self.cfg.min_y,self.cfg.max_y+1))
        self.not_avail_y = self.not_avail_y[~np.isin(self.not_avail_y,self.available_y)]
        mat_x = np.concatenate((rows[:, np.newaxis], cols[:, np.newaxis], vals_x[:, np.newaxis]), axis=1).astype(np.float32)
        mat_x_train = np.zeros((0,3),dtype = np.float32)
        for i in range(mat_x.shape[0]):
            if trainable_idx[int(mat_x[i][1])] == True:
                mat_x_train = np.concatenate([mat_x_train,mat_x[i].copy().reshape(1,-1)],axis=0)
        return mat_x_train, mat_x
    
    def convert_data_to_dataframe(self,mat_x):
        dataframe_x = self.spark.createDataFrame(np.float32(mat_x).tolist(), ["y_pts", "lane_id", "x_pts"])
        return dataframe_x

    def ALS(self,train_x, test_x,z_pts):
        # try:
            # als for x
            y_pts = np.array(range(self.cfg.min_y,self.cfg.max_y+1))
            rank = (2 if self.mode == 'straight' else 3)
            # model = ALS(rank = rank,
            #             maxIter = self.als_param['iterations'],
            #             userCol='lane_id',
            #             itemCol = 'y_pts',
            #             ratingCol = 'x_pts',
            #             nonnegative=True,
            #             # coldStartStrategy='drop'
            #             )
            # model = model.fit(test_x)
            # predictions = model.transform(test_x)
            # # 필요한 열을 선택하고 pandas DataFrame으로 변환
            # predictions_df = predictions.select('lane_id', 'y_pts', 'prediction').toPandas()
            # # unique y_pts와 lane_id 추출
            # unique_lane_ids = predictions_df['lane_id'].unique()
            # # y_pts와 lane_id의 크기에 맞는 결과 배열 생성
            # result_array = np.full((len(self.available_y), len(unique_lane_ids)), 0.0)

            # # y_pts와 lane_id 인덱스를 찾기 위한 맵 생성
            # y_pts_to_index = {y_pt: idx for idx, y_pt in enumerate(self.available_y)}
            # lane_id_to_index = {idx: idx for idx in range(len(unique_lane_ids))}

            # # 결과 배열에 값 채우기
            # for _, row in predictions_df.iterrows():
            #     y_pt_idx = y_pts_to_index[row['y_pts']]
            #     lane_id_idx = lane_id_to_index[row['lane_id']]
            #     result_array[y_pt_idx, lane_id_idx] = row['prediction']

            
            model = ALS.train(test_x,
                          rank=rank,
                          iterations=self.als_param['iterations'],
                          nonnegative=True
                          )

            self.model = model
            U_pred = model.userFeatures()
            V_pred = model.productFeatures()
            U_pred = np.asarray([y for x, y in sorted(U_pred.collect())])
            V_pred = np.asarray([y for x, y in sorted(V_pred.collect())])
            result_array = np.array(U_pred.dot(V_pred.T))
            # 결과 출력 또는 반환
            self.M_pred_x = np.zeros((self.cfg.max_y+1,z_pts.shape[1]))
            self.M_pred_x[self.available_y,:] = np.nan_to_num(result_array,nan=0.0)
            for idx in range(self.M_pred_x.shape[1]):
                f_x = interp1d(self.available_y,self.M_pred_x[self.available_y,idx],fill_value='extrapolate')
                new_x = f_x(self.not_avail_y)
                self.M_pred_x[self.not_avail_y,idx] = new_x
            self.M_pred_x = self.M_pred_x[y_pts,:].reshape(len(y_pts),-1)
            # linear interpolation/extrapolation for z
            self.M_pred_z = np.zeros((y_pts.shape[0],0))
            for i in range(z_pts.shape[1]):
                temp = z_pts[:,i]
                f_z = interp1d(y_pts[temp != 0], temp[temp != 0],bounds_error=False, fill_value = (temp[temp != 0][0],temp[temp != 0][-1]), kind='linear')
                z_new = f_z(y_pts).reshape(-1,1)
                self.M_pred_z = np.concatenate([self.M_pred_z,z_new],axis = 1)

            # # als for z
            # model = ALS.train(self.dataframe_z,
            #                 rank=rank,
            #                 iterations=self.als_param['iterations'],
            #                 lambda_=self.als_param['lambda'])

            # self.model = model
            # U_pred = model.userFeatures()
            # V_pred = model.productFeatures()
            # U_pred = np.asarray([y for x, y in sorted(U_pred.collect())])
            # V_pred = np.asarray([y for x, y in sorted(V_pred.collect())])
            # self.M_pred_z = np.array(U_pred.dot(V_pred.T))
            print(self.M_pred_x.shape,self.M_pred_z.shape)
            # self.sc.stop()
        # except:print('error')
