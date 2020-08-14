# ml-tools

# TODO
* Feature 
    * FEの便利関数いろいろ
* 後段は1pipe 1sourceの方針に変更 
    * いろいろなモデルの学習＋推論＋Feature Importance + サブミットをできるスクリプトを作る
        * LGBM, XGBoost, CatBoost, Scikit-learn, NN, NGBoostなど...
    * スタッキング用スクリプト（基本的には上と同様）
    * additional option
        * 疑似ラベル
        * (残差学習？)
* フォルダ構成
```
|features(FEの結果はここに溜まっていく)
|input(最初のデータ)
|outputs(モデルを学習した結果はここに溜まっていく)
|src(ここで作業)
    |__speeder(便利ツール入れ)
    |__config(configファイル入れ)
    |__fe(特徴量エンジニアリングの作業スクリプト)
    |__model（モデリングの作業スクリプト）
```
# 設計方針

## Feature
* 色々な特徴量作成に役立つ関数が入っている
* 特徴抽出に集中できるデコレータが入っている（保存とかはやってくれる）


## Modeling
* 抽象クラスに基づいてモデルを定義すると色々な学習方法 + 推論 + Feature Importance + サブミットまでやってくれる
* スタッキングを簡単にできる