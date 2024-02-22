from flask import Flask, render_template, request , jsonify
import time
import os
from flask_restful import Api
from my_tools.index import index_one,index_one_faiss
from my_tools.search import Search,SearchFaiss


app = Flask(__name__)
#general parameters
UPLOAD_FOLDER = 'static/img'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Index route
@app.route('/')
def index():

    return render_template('main.html',display_detection='static/img/img1.jpg' )

@app.route('/upload', methods=['POST'])
def upload():
    # Saving the Uploaded image in the Upload folder
    file = request.files['image']
    sort = request.form.get("vehi")

    new_file_name = str(
        "img1" + '.jpg'
    )
    file.save(os.path.join(
            app.config['UPLOAD_FOLDER'],new_file_name
        )
    )

    # Trích xuất vectơ đối tượng từ các hình ảnh đã tải lên và thêm vectơ này vào cơ sở dữ liệu của chúng tôi
    # features = index_one(str(UPLOAD_FOLDER + '/' + new_file_name) )
    features = index_one_faiss(str(UPLOAD_FOLDER + '/' + new_file_name))

    # So sánh và sắp xếp các tính năng của hình ảnh đã tải lên với các tính năng của hình ảnh calulcated ngoại tuyến

    # searcher = Search('my_tools/lfcnn/filecsv/lfcnn_pca_nn.csv')
    # results = searcher.search(features)
    # print(results)
    searcher = SearchFaiss('lfcnn_pca_nn2.csv')
    results = searcher.search(features)


    RESULTS_LIST = list()
    for (pathImage, score) in results.items():
        RESULTS_LIST.append(
            {"image": str(pathImage), "score": str(score)}
        )
    print(RESULTS_LIST)
    print(sort)


    #returning the search results
    return jsonify(RESULTS_LIST)

if __name__ == '__main__':
    app.run(debug=True)