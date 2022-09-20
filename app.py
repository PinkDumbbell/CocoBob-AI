from flask import Flask,request,jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

data = pd.read_csv('firstDB_Backup.csv', low_memory=False)

remove_string_data = data[data.columns.difference(['product_id','category','code','kcal_per_kg','name','thumbnail','product_image','product_detail_image','description','brand'])]

cosine_sim = cosine_similarity(remove_string_data, remove_string_data)

indices = pd.Series(data.index, index=data['product_id']).drop_duplicates()

def get_recommendations(prouct_id, cosine_sim=cosine_sim):
    # 선택한 상품의 id로부터 해당되는 인덱스를 받아옵니다. 이제 선택한 상품을 가지고 연산할 수 있습니다.
    idx = indices[prouct_id]

    # 모든 상품에 대해서 해당 다른 상품와의 유사도를 구합니다.
    sim_scores = list(enumerate(cosine_sim[idx]))
    # 유사도에 따라 영화들을 정렬합니다.
    sim_scores = sorted(sim_scores,key = lambda x:(x[1]), reverse=True)

    # 가장 유사한 10개의 상품을 받아옵니다.
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 상품의 인덱스를 받아옵니다.
    product_indices = [i[0] for i in sim_scores]
    # 가장 유사한 10개의 상품을 리턴합니다.
    return data[['product_id','name']].iloc[product_indices]

@app.route("/related",methods=['GET'])
def get_related_products():
    product_id = int(request.args.get('productId'))
    data = get_recommendations(product_id,cosine_sim)
    return jsonify(data.to_dict('records'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
