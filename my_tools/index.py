import os
import pandas as pd
import cv2
import numpy as np
from my_tools.lfcnn.features_cbir_run import cnn_one_image,feature_extraction_img_809,norm_minmax_feature,loadthamso
import pickle
from sklearn.decomposition import PCA
import  csv
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD


b_loaded_cnn = False



def dimension_reduction(file_index):
    df = pd.read_csv(file_index)
    X = df[df.columns[1:]]
    imgpath = df['label']
    print(imgpath.shape)
    pca = PCA(n_components=500)
    pcalfCNN = pca.fit_transform(X)

    pickle.dump(pca, open("model/pca_new500.pkl", "wb"))
    dflfcnn = pd.DataFrame(pcalfCNN)
    dflfcnn['imgpath'] =imgpath
    dflfcnn.to_csv("lfcnn/filecsv/lfcnn_pca_nn500.csv",index=False)
    print("done!")

def dimension_reduction_TruncatedSVD(file_index):
    df = pd.read_csv(file_index)
    X = df[df.columns[1:]]
    imgpath = df['label']

    ICA = TruncatedSVD(n_components=4200, random_state=226)
    X = ICA.fit_transform(X)

    pickle.dump(ICA, open("model/SVD_new.pkl", "wb"))
    dflfcnn = pd.DataFrame(X)
    dflfcnn['imgpath'] =imgpath
    dflfcnn.to_csv("lfcnn/filecsv/lfcnn_SVD_n.csv",index=False)
    print("done!")


def index_one(imagepath):
    pca_reload = pickle.load(open("my_tools/model/pca_newn.pkl", 'rb'))
    # ica_reload = pickle.load(open("my_tools/model/SVD_new.pkl", 'rb'))

    global vCNN
    v_min, v_max, v_mean, v_var, v_std = loadthamso('my_tools/lfcnn/filecsv/thongkenew.csv')
    I = cv2.imread(imagepath)
    if I is not None:
        vCNN = cnn_one_image(I)

    vlf = feature_extraction_img_809(I)

    vlf_n=norm_minmax_feature(np.asarray([vlf]),v_min,v_max)[0]

    vlf_cnn= np.concatenate((vlf_n,vCNN), axis=None)

    result_new = pca_reload.transform(vlf_cnn.reshape(1, -1))[0]

    return result_new

def index_one_faiss(imagepath):
    # 3800 feutur
    # pca_reload = pickle.load(open("my_tools/model/pca_newn.pkl", 'rb'))

    # 2600 fetur
    pca_reload = pickle.load(open("my_tools/model/pca_new500.pkl", 'rb'))

    global vCNN
    v_min, v_max, v_mean, v_var, v_std = loadthamso('my_tools/lfcnn/filecsv/thongkenew.csv')
    I = cv2.imread(imagepath)
    if I is not None:
        vCNN = cnn_one_image(I)

    vlf = feature_extraction_img_809(I)

    vlf_n=norm_minmax_feature(np.asarray([vlf]),v_min,v_max)[0]

    vlf_cnn= np.concatenate((vlf_n,vCNN), axis=None)

    result_new = pca_reload.transform(vlf_cnn.reshape(1, -1))[0]
    query = np.reshape(result_new, (-1, 500))
    query = np.float32(query)
    return query

if __name__ == '__main__':

   # giảm chiều dữ liệu
   #  dimension_reduction_thongke('lfcnn/filecsv/thongkenewne.csv')
    dimension_reduction('lfcnn/filecsv/lfcnnnew.csv')
#     # dimension_reduction_FastICA('lfcnn/filecsv/lfcnn.csv')
#     dimension_reduction_TruncatedSVD('lfcnn/filecsv/lfcnnnewne.csv')

    # # test ảnh

    # print(index_one('D:\\HocTapEPU\\CBIR\\CBIRThi\\static\\images_y\\0ACIIO7722UH.jpg'))



