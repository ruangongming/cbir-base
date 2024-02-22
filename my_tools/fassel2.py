import faiss
import numpy as np
import pandas as pd
import csv

import pickle
import cv2
from my_tools.lfcnn.features_cbir_run import cnn_one_image,feature_extraction_img_809,norm_minmax_feature,loadthamso
def faiss_feuter(feuter,query_img):

    index = faiss.IndexFlatL2(3800)
    descriptors = np.vstack(feuter)
    index.add(descriptors)
    distance, indices = index.search(query_img,20)
    return distance,indices





if __name__ == '__main__':
    df = pd.read_csv('lfcnn_pca_nn.csv')
    X = df[df.columns[:3800]]
    y=df['imgpath']
    des = X.to_numpy()
    imgpath = y.to_numpy()

    des32 = np.float32(des)
    l_img_path = list(imgpath)

    desto = []
    for i in range(len(des)):
        desto.append(np.reshape(des32[i], (-1, 3800)))


    pca_reload = pickle.load(open("model/pca_newn.pkl", 'rb'))

    global vCNN
    v_min, v_max, v_mean, v_var, v_std = loadthamso('lfcnn/filecsv/thongkenew.csv')
    I = cv2.imread("D:\\HocTapEPU\\CBIR\\CBIRThi\\static\\images_y\\1FDYXTWLGMRH.jpg")
    if I is not None:
        vCNN = cnn_one_image(I)

    vlf = feature_extraction_img_809(I)

    vlf_n = norm_minmax_feature(np.asarray([vlf]), v_min, v_max)[0]

    vlf_cnn = np.concatenate((vlf_n, vCNN), axis=None)

    result_new = pca_reload.transform(vlf_cnn.reshape(1, -1))[0]


    query = np.reshape(result_new, (-1, 3800))
    query=np.float32(query)

    distance,indices = faiss_feuter(desto, query)
    dis = distance[0]
    ind = indices[0]

    img_inds = []
    for file_index in (ind):
        print(file_index)
        img_inds.append(l_img_path[file_index])
    res = dict(zip(img_inds,dis))

    print(res)







