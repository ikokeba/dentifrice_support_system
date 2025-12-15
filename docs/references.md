# リソース・参考資料

本ドキュメントでは、Android歯磨き認識モデル統合プロジェクトに関連する有用なリソースと参考資料をまとめています。

## 目次

1. [TensorFlow公式ドキュメント](#1-tensorflow公式ドキュメント)
2. [TensorFlow Hubモデル](#2-tensorflow-hubモデル)
3. [Android開発リソース](#3-android開発リソース)
4. [データセット](#4-データセット)
5. [GitHubリポジトリ](#5-githubリポジトリ)
6. [論文・研究資料](#6-論文研究資料)
7. [コミュニティ・フォーラム](#7-コミュニティフォーラム)
8. [ツール・ユーティリティ](#8-ツールユーティリティ)

## 1. TensorFlow公式ドキュメント

### 1.1 TensorFlow Lite

- **公式サイト**: https://www.tensorflow.org/lite
- **ガイド**: https://www.tensorflow.org/lite/guide
- **APIリファレンス**: https://www.tensorflow.org/lite/api_docs
- **Androidガイド**: https://www.tensorflow.org/lite/android
- **モデル最適化**: https://www.tensorflow.org/lite/performance/model_optimization

### 1.2 TensorFlow Hub

- **公式サイト**: https://tfhub.dev/
- **I3Dモデル**: https://tfhub.dev/deepmind/i3d-kinetics-400/1
- **モデル検索**: https://tfhub.dev/s?q=kinetics
- **ドキュメント**: https://www.tensorflow.org/hub

### 1.3 TensorFlow Model Optimization

- **公式ガイド**: https://www.tensorflow.org/model_optimization
- **量子化ガイド**: https://www.tensorflow.org/model_optimization/guide/quantization/post_training
- **プルーニングガイド**: https://www.tensorflow.org/model_optimization/guide/pruning

### 1.4 TensorFlow Lite Model Maker

- **公式ガイド**: https://www.tensorflow.org/lite/models/modify/model_maker
- **ビデオ分類**: https://www.tensorflow.org/lite/models/modify/model_maker/video_classification

## 2. TensorFlow Hubモデル

### 2.1 I3Dモデル

- **Kinetics-400**: https://tfhub.dev/deepmind/i3d-kinetics-400/1
- **Kinetics-600**: https://tfhub.dev/deepmind/i3d-kinetics-600/1
- **Kinetics-700**: https://tfhub.dev/deepmind/i3d-kinetics-700/1

### 2.2 その他の動画認識モデル

- **MobileNet V3**: https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5
- **EfficientNet**: https://tfhub.dev/tensorflow/efficientnet/b0/classification/1

### 2.3 モデルの比較

- **TensorFlow Hubモデル一覧**: https://tfhub.dev/s?module-type=image-classification
- **モデル性能比較**: https://www.tensorflow.org/hub/common_signatures/images

## 3. Android開発リソース

### 3.1 CameraX

- **公式ドキュメント**: https://developer.android.com/training/camerax
- **APIリファレンス**: https://developer.android.com/reference/androidx/camera/core/package-summary
- **サンプルコード**: https://github.com/android/camera-samples
- **ガイド**: https://developer.android.com/training/camerax/getting-started

### 3.2 TensorFlow Lite Android

- **公式ガイド**: https://www.tensorflow.org/lite/android
- **サンプルアプリ**: https://github.com/tensorflow/examples/tree/master/lite/examples
- **ビデオ分類サンプル**: https://github.com/tensorflow/examples/tree/master/lite/examples/video_classification/android

### 3.3 Android開発全般

- **Android Developers**: https://developer.android.com/
- **Kotlin公式**: https://kotlinlang.org/
- **Android Studio**: https://developer.android.com/studio

## 4. データセット

### 4.1 Kineticsデータセット

- **公式サイト**: https://deepmind.com/research/open-source/kinetics
- **論文**: "The Kinetics Human Action Video Dataset" (arXiv:1705.06950)
- **ダウンロード**: https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics
- **ラベルリスト**: https://github.com/activitynet/ActivityNet/blob/master/Crawler/Kinetics/readme.md

### 4.2 NTU RGB+Dデータセット

- **公式サイト**: https://rose1.ntu.edu.sg/datasets/actionrecognition.asp
- **論文**: "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis" (CVPR 2016)
- **ダウンロード**: 公式サイトから申請が必要

### 4.3 その他の行動認識データセット

- **UCF-101**: https://www.crcv.ucf.edu/data/UCF101.php
- **HMDB51**: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
- **Something-Something V2**: https://20bn.com/datasets/something-something

## 5. GitHubリポジトリ

### 5.1 TensorFlow公式リポジトリ

- **TensorFlow**: https://github.com/tensorflow/tensorflow
- **TensorFlow Lite**: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite
- **TensorFlow Examples**: https://github.com/tensorflow/examples
- **TensorFlow Hub Models**: https://github.com/tensorflow/hub

### 5.2 I3D実装

- **deepmind/kinetics-i3d**: https://github.com/deepmind/kinetics-i3d
- **tensorflow/models (I3D)**: https://github.com/tensorflow/models/tree/master/research/slim/nets

### 5.3 NTU RGB+D実装

- **shahroudy/NTURGB-D**: https://github.com/shahroudy/NTURGB-D
- **各種実装**: GitHubで"NTU RGB+D"を検索

### 5.4 Androidサンプル

- **tensorflow/examples (Android)**: https://github.com/tensorflow/examples/tree/master/lite/examples
- **android/camera-samples**: https://github.com/android/camera-samples
- **tensorflow-lite-video-classification**: GitHubで検索

### 5.5 モデル変換ツール

- **onnx2tf**: https://github.com/onnx/onnx-tensorflow
- **onnx-tensorflow**: https://github.com/onnx/onnx-tensorflow

## 6. 論文・研究資料

### 6.1 I3D関連

- **Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset** (CVPR 2017)
  - 著者: Joao Carreira, Andrew Zisserman
  - arXiv: https://arxiv.org/abs/1705.07750
  - 公式サイト: https://deepmind.com/research/open-source/kinetics

### 6.2 NTU RGB+D関連

- **NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis** (CVPR 2016)
  - 著者: Amir Shahroudy, Jun Liu, Tian-Tsong Ng, Gang Wang
  - 論文: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf

- **NTU RGB+D 120: A Large-Scale Benchmark for 3D Human Activity Understanding** (TPAMI 2019)
  - 著者: Jun Liu, Amir Shahroudy, Mauricio Perez, Gang Wang, Ling-Yu Duan, Alex C. Kot
  - 論文: https://ieeexplore.ieee.org/document/8809919

### 6.3 動画認識全般

- **Two-Stream Convolutional Networks for Action Recognition in Videos** (NIPS 2014)
  - 著者: Karen Simonyan, Andrew Zisserman
  - arXiv: https://arxiv.org/abs/1406.2199

- **Temporal Segment Networks: Towards Good Practices for Deep Action Recognition** (ECCV 2016)
  - 著者: Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool
  - arXiv: https://arxiv.org/abs/1608.00859

## 7. コミュニティ・フォーラム

### 7.1 TensorFlowコミュニティ

- **TensorFlow Forum**: https://discuss.tensorflow.org/
- **Stack Overflow (TensorFlowタグ)**: https://stackoverflow.com/questions/tagged/tensorflow
- **Reddit (r/tensorflow)**: https://www.reddit.com/r/tensorflow/
- **GitHub Discussions**: https://github.com/tensorflow/tensorflow/discussions

### 7.2 Android開発コミュニティ

- **Stack Overflow (Androidタグ)**: https://stackoverflow.com/questions/tagged/android
- **Reddit (r/androiddev)**: https://www.reddit.com/r/androiddev/
- **Android Developers Community**: https://developer.android.com/community

### 7.3 機械学習コミュニティ

- **Papers with Code**: https://paperswithcode.com/
- **arXiv (Computer Vision)**: https://arxiv.org/list/cs.CV/recent
- **r/MachineLearning**: https://www.reddit.com/r/MachineLearning/

## 8. ツール・ユーティリティ

### 8.1 モデル変換ツール

- **onnx2tf**: https://github.com/onnx/onnx-tensorflow
  - ONNXモデルをTensorFlow形式に変換

- **onnx-tf**: https://github.com/onnx/onnx-tensorflow
  - ONNXからTensorFlowへの変換ツール

- **TensorFlow Lite Converter**: 
  - TensorFlowに同梱
  - `tf.lite.TFLiteConverter`を使用

### 8.2 モデル可視化ツール

- **Netron**: https://github.com/lutzroeder/netron
  - モデル構造を可視化（TFLite対応）

- **TensorBoard**: https://www.tensorflow.org/tensorboard
  - TensorFlowモデルの可視化

### 8.3 パフォーマンス測定ツール

- **Android Profiler**: 
  - Android Studioに同梱
  - メモリ、CPU、ネットワークのプロファイリング

- **TensorFlow Lite Benchmark Tool**:
  - https://www.tensorflow.org/lite/performance/measurement
  - モデルの推論速度を測定

### 8.4 データ前処理ツール

- **FFmpeg**: https://ffmpeg.org/
  - ビデオファイルの処理・変換

- **OpenCV**: https://opencv.org/
  - 画像・動画処理ライブラリ
  - Python: `pip install opencv-python`
  - Android: OpenCV for Android SDK

### 8.5 ラベル管理ツール

- **Label Studio**: https://labelstud.io/
  - データセットのラベリングツール

- **VGG Image Annotator (VIA)**: https://www.robots.ox.ac.uk/~vgg/software/via/
  - 画像・動画のアノテーションツール

## 9. 学習リソース

### 9.1 オンラインコース

- **TensorFlow Developer Certificate**: https://www.tensorflow.org/certificate
- **Coursera - TensorFlow in Practice**: https://www.coursera.org/specializations/tensorflow-in-practice
- **Udacity - Android Developer Nanodegree**: https://www.udacity.com/course/android-developer-nanodegree--nd801

### 9.2 チュートリアル・ブログ

- **TensorFlow Blog**: https://blog.tensorflow.org/
- **Android Developers Blog**: https://android-developers.googleblog.com/
- **Towards Data Science**: https://towardsdatascience.com/ (Medium)

### 9.3 YouTubeチャンネル

- **TensorFlow**: https://www.youtube.com/c/TensorFlow
- **Android Developers**: https://www.youtube.com/c/AndroidDevelopers
- **3Blue1Brown**: https://www.youtube.com/c/3blue1brown (機械学習の基礎)

## 10. 書籍

### 10.1 TensorFlow関連

- **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** by Aurélien Géron
- **"Deep Learning with Python"** by François Chollet

### 10.2 Android開発関連

- **"Android Programming: The Big Nerd Ranch Guide"** by Bill Phillips, Chris Stewart, Kristin Marsicano
- **"Kotlin in Action"** by Dmitry Jemerov, Svetlana Isakova

### 10.3 コンピュータビジョン関連

- **"Computer Vision: Algorithms and Applications"** by Richard Szeliski
- **"Deep Learning for Computer Vision"** by Rajalingappaa Shanmugamani

## 11. その他の有用なリソース

### 11.1 モデルハブ・マーケットプレイス

- **TensorFlow Hub**: https://tfhub.dev/
- **Hugging Face**: https://huggingface.co/
- **Model Zoo**: https://modelzoo.co/

### 11.2 ベンチマーク・評価

- **Papers with Code Leaderboards**: https://paperswithcode.com/sota
- **Model Database**: https://modelzoo.co/

### 11.3 ニュースレター・メールリスト

- **TensorFlow Newsletter**: https://www.tensorflow.org/community/newsletter
- **Android Developers Newsletter**: https://developer.android.com/newsletter

## 12. ローカルリソース

### 12.1 プロジェクト内ドキュメント

- **設計ドキュメント**: `docs/design_document.md`
- **実装ガイド**: `docs/implementation_guide.md`
- **トラブルシューティング**: `docs/troubleshooting.md`

### 12.2 プロジェクト構造

```
dentifrice_support_system/
├── docs/
│   ├── design_document.md
│   ├── implementation_guide.md
│   ├── troubleshooting.md
│   └── references.md
├── models/
│   ├── saved_model/
│   └── tflite/
└── scripts/
    ├── download_i3d_model.py
    └── convert_to_tflite.py
```

## 13. 更新履歴

- **2024-12-15**: 初版作成
- リソースは定期的に更新されるため、最新情報は各公式サイトを確認してください

## 14. 貢献・フィードバック

リソースの追加や修正の提案がある場合は、プロジェクトのIssueトラッカーまたはプルリクエストでお知らせください。

---

**注意**: リンクは2024年12月時点のものです。リンクが切れている場合は、各公式サイトの検索機能を使用してください。

