# トラブルシューティングガイド

本ガイドでは、モデル変換およびAndroidアプリ統合時に発生する可能性のある問題とその解決方法を説明します。

## 目次

1. [モデル変換時の問題](#1-モデル変換時の問題)
2. [Android統合時の問題](#2-android統合時の問題)
3. [パフォーマンス問題](#3-パフォーマンス問題)
4. [その他の問題](#4-その他の問題)

## 1. モデル変換時の問題

### 1.1 TensorFlow Hubからのモデルダウンロードが失敗する

**症状**:
```
Error: Failed to download model from TensorFlow Hub
```

**原因**:
- インターネット接続の問題
- TensorFlow Hubのサーバーが一時的に利用不可
- ファイアウォールやプロキシの設定

**解決方法**:

1. **インターネット接続を確認**:
   ```powershell
   ping tfhub.dev
   ```

2. **プロキシ設定を確認**（企業ネットワークの場合）:
   ```python
   import os
   os.environ['HTTP_PROXY'] = 'http://proxy.example.com:8080'
   os.environ['HTTPS_PROXY'] = 'http://proxy.example.com:8080'
   ```

3. **リトライ**:
   ネットワークが不安定な場合、数回リトライすると成功する場合があります。

4. **代替方法**: モデルを手動でダウンロードし、ローカルパスから読み込む

### 1.2 SavedModelへの変換が失敗する

**症状**:
```
Error: Signature 'default' not found
```

**原因**:
- モデルのシグネチャ名が異なる
- モデルの構造が期待と異なる

**解決方法**:

1. **利用可能なシグネチャを確認**:
   ```python
   import tensorflow_hub as hub
   model = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1")
   print("Available signatures:", list(model.signatures.keys()))
   ```

2. **正しいシグネチャを使用**:
   ```python
   # 例: 'video'シグネチャが存在する場合
   model_signature = model.signatures['video']
   ```

3. **モデルの構造を確認**:
   ```python
   # モデルの詳細を確認
   print(model)
   ```

### 1.3 TFLite変換時にエラーが発生する

**症状**:
```
Error: Some ops are not supported by the standard TensorFlow Lite runtime
```

**原因**:
- I3DモデルにTFLite標準でサポートされていないopsが含まれている

**解決方法**:

1. **SELECT_TF_OPSを有効化**:
   ```python
   converter.target_spec.supported_ops = [
       tf.lite.OpsSet.TFLITE_BUILTINS,
       tf.lite.OpsSet.SELECT_TF_OPS
   ]
   ```

2. **Androidアプリに依存関係を追加**:
   ```gradle
   implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0'
   ```

3. **変換オプションを調整**:
   ```python
   # 一部のopsを除外して変換を試す
   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
   ```

### 1.4 量子化変換が失敗する

**症状**:
```
Error: Representative dataset is required for quantization
```

**原因**:
- int8量子化には代表データセットが必要

**解決方法**:

1. **代表データセットを提供**:
   ```python
   def representative_dataset():
       # 実際のデータサンプルを使用（推奨）
       for data in your_dataset:
           yield [data]
       
       # またはダミーデータを使用（精度が低下する可能性）
       for _ in range(100):
           yield [tf.random.normal((1, 16, 224, 224, 3))]
   
   converter.representative_dataset = representative_dataset
   ```

2. **float16量子化を使用**（代表データセット不要）:
   ```python
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   converter.target_spec.supported_types = [tf.float16]
   ```

### 1.5 変換後のモデルサイズが大きすぎる

**症状**:
- モデルファイルが100MB以上になる
- Androidアプリのサイズが大きくなる

**解決方法**:

1. **int8量子化を適用**:
   ```python
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   converter.representative_dataset = representative_dataset
   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
   converter.inference_input_type = tf.int8
   converter.inference_output_type = tf.int8
   ```

2. **モデルプルーニングを検討**（高度な手法）:
   - TensorFlow Model Optimization Toolkitを使用
   - 重要度の低いニューロンを削除

3. **より軽量なモデルを検討**:
   - MobileNetベースのモデル
   - NTU RGB+Dの軽量版

## 2. Android統合時の問題

### 2.1 モデルファイルが読み込めない

**症状**:
```
Error: Failed to load model from assets
java.io.FileNotFoundException: i3d.tflite
```

**原因**:
- ファイルパスが間違っている
- assetsフォルダにファイルが存在しない
- ファイル名の大文字小文字が間違っている

**解決方法**:

1. **ファイルパスを確認**:
   ```
   app/src/main/assets/i3d.tflite
   ```

2. **ファイル名の大文字小文字を確認**:
   - Androidは大文字小文字を区別します
   - `i3d.tflite`と`I3D.tflite`は異なります

3. **ビルドをクリーン**:
   ```
   Build → Clean Project
   Build → Rebuild Project
   ```

4. **ファイルが正しくコピーされているか確認**:
   ```kotlin
   // デバッグ用コード
   val assets = context.assets.list("")
   Log.d("Assets", "Files in assets: ${assets?.joinToString()}")
   ```

### 2.2 Interpreterの初期化が失敗する

**症状**:
```
Error: Failed to create Interpreter
java.lang.IllegalArgumentException: Model is not a valid FlatBuffer
```

**原因**:
- モデルファイルが破損している
- モデルファイルの形式が正しくない
- ファイルが圧縮されている

**解決方法**:

1. **aaptOptionsを確認**:
   ```gradle
   android {
       aaptOptions {
           noCompress "tflite"
       }
   }
   ```

2. **モデルファイルを再変換**:
   Pythonスクリプトでモデルを再変換し、正しく保存されているか確認

3. **モデルファイルの検証**:
   ```python
   import tensorflow as tf
   interpreter = tf.lite.Interpreter(model_path="i3d.tflite")
   interpreter.allocate_tensors()
   print("Model is valid")
   ```

### 2.3 推論時の入力形状エラー

**症状**:
```
Error: Input tensor shape mismatch
Expected: [1, 16, 224, 224, 3], Got: [1, 15, 224, 224, 3]
```

**原因**:
- フレーム数が16に満たない
- 入力テンソルの形状が間違っている

**解決方法**:

1. **フレーム数を確認**:
   ```kotlin
   if (frames.size != 16) {
       Log.w("Analyzer", "Frame count: ${frames.size}, expected 16")
       return
   }
   ```

2. **フレームバッファリングを修正**:
   ```kotlin
   // 16フレームになるまで待つ
   if (frameBuffer.size < bufferSize) {
       return
   }
   ```

3. **入力テンソルの形状を確認**:
   ```kotlin
   val inputDetails = interpreter.getInputDetails()
   Log.d("Input", "Shape: ${inputDetails[0]['shape']}")
   ```

### 2.4 CameraXが起動しない

**症状**:
- カメラプレビューが表示されない
- カメラ権限エラー

**原因**:
- カメラ権限が許可されていない
- CameraXの設定が間違っている
- デバイスにカメラが存在しない

**解決方法**:

1. **権限を確認**:
   ```kotlin
   if (ContextCompat.checkSelfPermission(
           this,
           Manifest.permission.CAMERA
       ) != PackageManager.PERMISSION_GRANTED
   ) {
       // 権限をリクエスト
   }
   ```

2. **AndroidManifest.xmlを確認**:
   ```xml
   <uses-permission android:name="android.permission.CAMERA" />
   <uses-feature android:name="android.hardware.camera" android:required="true" />
   ```

3. **CameraXのバージョンを確認**:
   ```gradle
   def camerax_version = "1.3.0"
   implementation "androidx.camera:camera-core:${camerax_version}"
   ```

4. **エミュレーターの場合**:
   - エミュレーターにカメラが設定されているか確認
   - 実機でテストすることを推奨

### 2.5 推論が遅すぎる

**症状**:
- フレームレートが1fps以下
- UIがフリーズする

**原因**:
- 推論がメインスレッドで実行されている
- NNAPIが有効になっていない
- モデルが大きすぎる

**解決方法**:

1. **バックグラウンドスレッドで推論**:
   ```kotlin
   cameraExecutor = Executors.newSingleThreadExecutor()
   imageAnalysis.setAnalyzer(cameraExecutor, analyzer)
   ```

2. **NNAPIを有効化**:
   ```kotlin
   val options = Interpreter.Options().apply {
       setUseNNAPI(true)
   }
   ```

3. **フレームスキッピング**:
   ```kotlin
   private var isProcessing = false
   
   override fun analyze(imageProxy: ImageProxy) {
       if (isProcessing) {
           imageProxy.close()
           return
       }
       isProcessing = true
       // 処理
       isProcessing = false
   }
   ```

4. **推論頻度を下げる**:
   ```kotlin
   private var frameCount = 0
   
   override fun analyze(imageProxy: ImageProxy) {
       frameCount++
       if (frameCount % 30 != 0) { // 30フレームに1回のみ推論
           imageProxy.close()
           return
       }
       // 処理
   }
   ```

## 3. パフォーマンス問題

### 3.1 メモリリーク

**症状**:
- アプリのメモリ使用量が増え続ける
- 長時間実行するとクラッシュする

**原因**:
- Bitmapが適切にリサイクルされていない
- ImageProxyが適切にcloseされていない
- Interpreterが複数作成されている

**解決方法**:

1. **ImageProxyを確実にclose**:
   ```kotlin
   override fun analyze(imageProxy: ImageProxy) {
       try {
           // 処理
       } finally {
           imageProxy.close()
       }
   }
   ```

2. **Bitmapの再利用**:
   ```kotlin
   private val reusableBitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
   ```

3. **Interpreterをシングルトン化**:
   ```kotlin
   companion object {
       private var instance: Interpreter? = null
       
       fun getInstance(context: Context): Interpreter {
           if (instance == null) {
               instance = TFLiteModelLoader(context).loadModel("i3d.tflite")
           }
           return instance!!
       }
   }
   ```

### 3.2 バッテリー消費が高い

**症状**:
- バッテリーがすぐに減る
- デバイスが熱くなる

**原因**:
- 推論が頻繁に実行されている
- NNAPIが使用されていない
- CPUがフル稼働している

**解決方法**:

1. **推論頻度を下げる**:
   ```kotlin
   // 1秒に1回のみ推論
   private var lastInferenceTime = 0L
   
   override fun analyze(imageProxy: ImageProxy) {
       val currentTime = System.currentTimeMillis()
       if (currentTime - lastInferenceTime < 1000) {
           imageProxy.close()
           return
       }
       lastInferenceTime = currentTime
       // 処理
   }
   ```

2. **NNAPIを有効化**:
   ```kotlin
   setUseNNAPI(true)
   ```

3. **スレッド数を減らす**:
   ```kotlin
   setNumThreads(2) // 4から2に削減
   ```

4. **バックグラウンド時の処理停止**:
   ```kotlin
   override fun onPause() {
       super.onPause()
       cameraProvider.unbindAll()
   }
   
   override fun onResume() {
       super.onResume()
       setupCamera()
   }
   ```

### 3.3 推論精度が低い

**症状**:
- "brushing teeth"の確率が常に低い
- 他のアクションと誤認識される

**原因**:
- 量子化による精度低下
- 入力前処理が間違っている
- モデルが適切でない

**解決方法**:

1. **入力前処理を確認**:
   ```kotlin
   // 正規化が[0,1]になっているか確認
   inputBuffer.putFloat((pixel shr 16 and 0xFF) / 255.0f)
   ```

2. **量子化タイプを変更**:
   ```kotlin
   // int8からfloat16に変更
   // または非量子化モデルを使用
   ```

3. **フレーム数を確認**:
   ```kotlin
   // I3Dは16フレームを期待
   if (frames.size != 16) {
       return
   }
   ```

4. **ファインチューニングを検討**:
   - 子供の歯磨きデータでモデルを再訓練
   - TFLite Model Makerを使用

## 4. その他の問題

### 4.1 Android 15のカメラアクセス制限

**症状**:
- バックグラウンドでカメラが動作しない
- 権限エラーが発生する

**原因**:
- Android 15の新しいプライバシー機能

**解決方法**:

1. **フォアグラウンドでのみカメラを使用**:
   ```kotlin
   override fun onPause() {
       super.onPause()
       cameraProvider.unbindAll()
   }
   ```

2. **Foreground Serviceを使用**（必要に応じて）:
   - バックグラウンドでカメラを使用する場合はForeground Serviceが必要

### 4.2 ラベルファイルが見つからない

**症状**:
```
Error: File not found: kinetics_labels.txt
```

**解決方法**:

1. **ファイルパスを確認**:
   ```
   app/src/main/assets/kinetics_labels.txt
   ```

2. **ラベルファイルをダウンロード**:
   - GitHubリポジトリからKinetics-400のラベルリストを取得
   - または、モデルの出力インデックスから手動で作成

3. **ラベルファイルの形式を確認**:
   ```
   0: label_name
   1: another_label
   ...
   399: brushing_teeth
   ```

### 4.3 ビルドエラー

**症状**:
```
Error: Failed to resolve: org.tensorflow:tensorflow-lite:2.14.0
```

**解決方法**:

1. **リポジトリを確認**:
   ```gradle
   repositories {
       google()
       mavenCentral()
   }
   ```

2. **インターネット接続を確認**:
   - Gradleがリポジトリにアクセスできるか確認

3. **Gradleの同期**:
   ```
   File → Sync Project with Gradle Files
   ```

4. **キャッシュをクリア**:
   ```
   File → Invalidate Caches / Restart
   ```

### 4.4 デバッグ時のログが多すぎる

**症状**:
- Logcatに大量のログが出力される
- 重要なログが見つからない

**解決方法**:

1. **ログレベルを設定**:
   ```kotlin
   if (BuildConfig.DEBUG) {
       Log.d("YourTag", "Debug message")
   }
   ```

2. **Logcatフィルターを使用**:
   - Android StudioのLogcatでタグやレベルでフィルタリング

3. **カスタムロガーを使用**:
   ```kotlin
   object AppLogger {
       fun d(tag: String, message: String) {
           if (BuildConfig.DEBUG) {
               Log.d(tag, message)
           }
       }
   }
   ```

## 5. サポートとリソース

問題が解決しない場合:

1. **ログを確認**: Android StudioのLogcatでエラーメッセージを確認
2. **ドキュメントを参照**: `references.md`のリソースを確認
3. **コミュニティフォーラム**: TensorFlowやAndroidの公式フォーラムで質問
4. **GitHub Issues**: 関連プロジェクトのGitHub Issuesを検索

## 6. よくある質問（FAQ）

### Q: モデル変換にどのくらい時間がかかりますか？

A: モデルのサイズと量子化オプションによりますが、通常5-15分程度です。

### Q: Android Studioのバージョンは何が必要ですか？

A: Android Studio Hedgehog (2023.1.1)以降を推奨します。

### Q: エミュレーターで動作確認できますか？

A: 可能ですが、パフォーマンスが実機と異なる場合があります。実機でのテストを推奨します。

### Q: モデルをカスタマイズできますか？

A: はい、TFLite Model Makerを使用してファインチューニングが可能です。

### Q: コストはかかりますか？

A: すべて無料です。TensorFlow Hub、TensorFlow Lite、Android Studioはすべて無料で使用できます。

