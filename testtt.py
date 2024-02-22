from my_tools.index import index_one_faiss
from my_tools.search import SearchFaiss


features = index_one_faiss("D:\\HocTapEPU\\CBIR\\CBIRThi\\static\\images_y\\0ACIIO7722UH.jpg")
print(features)



searcher = SearchFaiss('lfcnn_pca_nn.csv')
results = searcher.search(features)



RESULTS_LIST = list()
for (pathImage,score ) in results.items():
    RESULTS_LIST.append(
        {"image": str(pathImage), "score": str(score)}
    )
print(RESULTS_LIST)