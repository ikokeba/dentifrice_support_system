# Android歯磨き認識モデル実装ガイド

本ガイドでは、設計ドキュメントに基づいて、実際にモデルを変換し、Androidアプリに統合するまでのステップバイステップの手順を説明します。

## 目次

1. [環境構築](#1-環境構築)
2. [モデル変換ステップ](#2-モデル変換ステップ)
3. [Android統合ステップ](#3-android統合ステップ)
4. [テスト・検証](#4-テスト検証)

## 1. 環境構築

### 1.1 Python環境構築（uv使用）

#### 1.1.1 uvのインストール

Windows環境でのuvインストール:

```powershell
# PowerShellで実行
irm https://astral.sh/uv/install.ps1 | iex
```

インストール後、新しいPowerShellセッションを開くか、環境変数をリロード:

```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

#### 1.1.2 プロジェクトディレクトリの作成

```powershell
# プロジェクトディレクトリに移動
cd C:\workspace\dentifrice_support_system

# 仮想環境を作成（.venv）
uv venv

# 仮想環境をアクティベート
.\.venv\Scripts\Activate.ps1
```

#### 1.1.3 依存関係のインストール

`requirements.txt`を作成:

```txt
tensorflow==2.14.0
tensorflow-hub==0.15.0
numpy==1.24.3
Pillow==10.0.0
```

インストール:

```powershell
uv pip install -r requirements.txt
```

または、uvのプロジェクト管理を使用:

```powershell
# pyproject.tomlを作成（後述）
uv pip install tensorflow==2.14.0 tensorflow-hub==0.15.0 numpy==1.24.3 Pillow==10.0.0
```

#### 1.1.4 動作確認

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import tensorflow_hub as hub; print('TensorFlow Hub OK')"
```

### 1.2 Android開発環境構築

#### 1.2.1 Android Studioのインストール

1. [Android Studio公式サイト](https://developer.android.com/studio)からダウンロード
2. インストーラーを実行し、標準設定でインストール
3. 初回起動時にSDKコンポーネントをダウンロード

#### 1.2.2 SDK要件

以下のSDKコンポーネントが必要です:

- **Android SDK Platform 34** (Android 14)
- **Android SDK Build-Tools 34.0.0**
- **Android SDK Platform-Tools**
- **NDK** (オプション、ネイティブ開発用)

Android StudioのSDK Managerからインストール:

```
Tools → SDK Manager → SDK Platforms → Android 14.0 (API 34) にチェック
Tools → SDK Manager → SDK Tools → Android SDK Build-Tools 34.0.0 にチェック
```

#### 1.2.3 必要なツール

- **Gradle**: Android Studioに同梱（バージョン8.1.0以上）
- **JDK**: Android Studioに同梱（JDK 17推奨）
- **Git**: バージョン管理用（オプション）

#### 1.2.4 環境変数の設定（オプション）

コマンドラインからAndroid開発ツールを使用する場合:

```powershell
# 環境変数に追加（PowerShell）
$env:ANDROID_HOME = "C:\Users\<ユーザー名>\AppData\Local\Android\Sdk"
$env:Path += ";$env:ANDROID_HOME\platform-tools;$env:ANDROID_HOME\tools"
```

## 2. モデル変換ステップ

### 2.1 I3Dモデル取得と変換

#### 2.1.1 作業ディレクトリの準備

```powershell
# プロジェクトルートで実行
mkdir models
mkdir models\saved_model
mkdir models\tflite
```

#### 2.1.2 TensorFlow Hubからのダウンロード

`scripts/download_i3d_model.py`を作成:

```python
"""
I3DモデルをTensorFlow Hubからダウンロードし、SavedModel形式に変換
"""
import tensorflow as tf
import tensorflow_hub as hub
import os

def download_and_save_i3d_model():
    """I3DモデルをダウンロードしてSavedModel形式で保存"""
    
    # モデルURL
    model_url = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
    
    print(f"モデルをダウンロード中: {model_url}")
    
    # モデルをロード
    model = hub.load(model_url)
    
    # シグネチャを確認
    print("利用可能なシグネチャ:", list(model.signatures.keys()))
    
    # デフォルトシグネチャを取得
    model_signature = model.signatures['default']
    
    # テスト入力を作成（I3Dの入力形状: [batch, frames, height, width, channels]）
    # I3Dは通常16フレームを入力とする
    test_input = tf.zeros((1, 16, 224, 224, 3), dtype=tf.float32)
    
    print("テスト推論を実行中...")
    # 推論を実行してモデルをトレース
    output = model_signature(test_input)
    print(f"出力形状: {output.shape}")
    
    # SavedModelとして保存
    saved_model_path = './models/saved_model/i3d_kinetics_400'
    print(f"SavedModelとして保存中: {saved_model_path}")
    
    tf.saved_model.save(model, saved_model_path)
    
    print("✓ SavedModelの保存が完了しました")
    return saved_model_path

if __name__ == "__main__":
    download_and_save_i3d_model()
```

実行:

```powershell
python scripts/download_i3d_model.py
```

**注意**: 初回実行時はモデルのダウンロードに時間がかかります（数分〜10分程度）。

#### 2.1.3 TFLite変換コード例

`scripts/convert_to_tflite.py`を作成:

```python
"""
SavedModelをTensorFlow Lite形式に変換
"""
import tensorflow as tf
import os

def convert_saved_model_to_tflite(
    saved_model_path: str,
    output_path: str,
    quantization: str = "int8"
):
    """
    SavedModelをTFLite形式に変換
    
    Args:
        saved_model_path: SavedModelのパス
        output_path: 出力TFLiteファイルのパス
        quantization: 量子化タイプ ("none", "float16", "int8")
    """
    
    print(f"SavedModelを読み込み中: {saved_model_path}")
    
    # TFLiteコンバーターを作成
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    
    # 量子化設定
    if quantization == "float16":
        print("float16量子化を適用中...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
    elif quantization == "int8":
        print("int8量子化を適用中...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # 代表データセットを提供（実際のデータがある場合は使用）
        def representative_dataset():
            for _ in range(100):
                yield [tf.random.normal((1, 16, 224, 224, 3), dtype=tf.float32)]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
    else:
        print("量子化なしで変換中...")
    
    # SELECT_TF_OPSを許可（I3Dに必要な場合）
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # TFLiteモデルに変換
    print("TFLiteモデルに変換中...")
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"変換エラー: {e}")
        print("SELECT_TF_OPSなしで再試行中...")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
    
    # ファイルに保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # ファイルサイズを表示
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ TFLiteモデルの保存が完了しました: {output_path}")
    print(f"  ファイルサイズ: {file_size_mb:.2f} MB")

def main():
    """メイン処理"""
    
    saved_model_path = "./models/saved_model/i3d_kinetics_400"
    
    # 複数の量子化オプションで変換
    quantization_options = [
        ("none", "i3d_float32.tflite"),
        ("float16", "i3d_float16.tflite"),
        ("int8", "i3d_int8.tflite")
    ]
    
    for quant_type, output_name in quantization_options:
        output_path = f"./models/tflite/{output_name}"
        print(f"\n{'='*60}")
        print(f"変換オプション: {quant_type}")
        print(f"{'='*60}")
        
        try:
            convert_saved_model_to_tflite(
                saved_model_path,
                output_path,
                quantization=quant_type
            )
        except Exception as e:
            print(f"✗ 変換失敗 ({quant_type}): {e}")
            continue

if __name__ == "__main__":
    main()
```

実行:

```powershell
python scripts/convert_to_tflite.py
```

#### 2.1.4 量子化設定の詳細

**int8量子化（推奨）**:
- サイズ: 約75-125MB（元の25-30%）
- 精度: わずかな低下（通常1-3%）
- 推論速度: 2-3倍高速化
- 用途: 本番環境

**float16量子化**:
- サイズ: 約150-250MB（元の50%）
- 精度: ほぼ維持
- 推論速度: 1.5-2倍高速化
- 用途: バランス重視

**非量子化（float32）**:
- サイズ: 約300-500MB
- 精度: 完全維持
- 推論速度: 標準
- 用途: 開発・検証

### 2.2 NTUモデル変換（PyTorch → ONNX → TFLite）

#### 2.2.1 PyTorchモデルの準備

NTU RGB+DベースのPyTorchモデルを取得（GitHubなどから）。

#### 2.2.2 PyTorch → ONNX変換

`scripts/pytorch_to_onnx.py`を作成:

```python
"""
PyTorchモデルをONNX形式に変換
"""
import torch
import torch.onnx

def convert_pytorch_to_onnx(
    pytorch_model_path: str,
    onnx_output_path: str,
    input_shape: tuple = (1, 3, 16, 224, 224)
):
    """
    PyTorchモデルをONNX形式に変換
    
    Args:
        pytorch_model_path: PyTorchモデルのパス（.pthファイル）
        onnx_output_path: 出力ONNXファイルのパス
        input_shape: 入力テンソルの形状 (batch, channels, frames, height, width)
    """
    
    # モデルをロード（モデルクラスの定義が必要）
    # model = YourNTUModel()
    # model.load_state_dict(torch.load(pytorch_model_path))
    # model.eval()
    
    # ダミー入力
    dummy_input = torch.randn(*input_shape)
    
    # ONNXにエクスポート
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'}
        },
        opset_version=11
    )
    
    print(f"✓ ONNXモデルの保存が完了しました: {onnx_output_path}")

if __name__ == "__main__":
    # 使用例（実際のモデルパスに置き換え）
    # convert_pytorch_to_onnx(
    #     "./models/pytorch/ntu_model.pth",
    #     "./models/onnx/ntu_model.onnx"
    # )
    pass
```

#### 2.2.3 ONNX → TFLite変換

`onnx2tf`ツールのインストール:

```powershell
uv pip install onnx2tf
```

`scripts/onnx_to_tflite.py`を作成:

```python
"""
ONNXモデルをTensorFlow Lite形式に変換
"""
import subprocess
import os
import tensorflow as tf

def convert_onnx_to_tflite(
    onnx_model_path: str,
    tflite_output_path: str
):
    """
    ONNXモデルをTFLite形式に変換
    
    Args:
        onnx_model_path: ONNXモデルのパス
        tflite_output_path: 出力TFLiteファイルのパス
    """
    
    # 中間ディレクトリ
    tf_model_dir = "./models/tf_intermediate"
    
    print(f"ONNX → TensorFlow変換中: {onnx_model_path}")
    
    # onnx2tfで変換
    result = subprocess.run(
        ["onnx2tf", "-i", onnx_model_path, "-o", tf_model_dir],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"✗ 変換エラー: {result.stderr}")
        return
    
    print("✓ TensorFlow形式への変換が完了しました")
    
    # TensorFlow → TFLite変換
    print("TensorFlow → TFLite変換中...")
    
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    os.makedirs(os.path.dirname(tflite_output_path), exist_ok=True)
    with open(tflite_output_path, 'wb') as f:
        f.write(tflite_model)
    
    file_size_mb = os.path.getsize(tflite_output_path) / (1024 * 1024)
    print(f"✓ TFLiteモデルの保存が完了しました: {tflite_output_path}")
    print(f"  ファイルサイズ: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    # 使用例
    # convert_onnx_to_tflite(
    #     "./models/onnx/ntu_model.onnx",
    #     "./models/tflite/ntu_model.tflite"
    # )
    pass
```

## 3. Android統合ステップ

### 3.1 プロジェクトセットアップ

#### 3.1.1 新しいAndroidプロジェクトの作成

1. Android Studioを起動
2. `File → New → New Project`
3. テンプレート: `Empty Activity`
4. 設定:
   - **Name**: ToothBrushingApp
   - **Package name**: com.example.toothbrushing
   - **Language**: Kotlin
   - **Minimum SDK**: API 24 (Android 7.0)
   - **Target SDK**: API 34 (Android 14)

#### 3.1.2 プロジェクト構造

```
app/
├── src/
│   └── main/
│       ├── assets/
│       │   ├── i3d_int8.tflite
│       │   └── kinetics_labels.txt
│       ├── java/com/example/toothbrushing/
│       │   ├── MainActivity.kt
│       │   ├── TFLiteModelLoader.kt
│       │   ├── ToothBrushingAnalyzer.kt
│       │   └── utils/
│       │       ├── ImageUtils.kt
│       │       └── LabelUtils.kt
│       └── res/
│           ├── layout/
│           │   └── activity_main.xml
│           └── values/
│               └── strings.xml
├── build.gradle
└── build.gradle (Project level)
```

### 3.2 依存関係追加

#### 3.2.1 アプリレベルのbuild.gradle

`app/build.gradle`を編集:

```gradle
plugins {
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
}

android {
    namespace 'com.example.toothbrushing'
    compileSdk 34

    defaultConfig {
        applicationId "com.example.toothbrushing"
        minSdk 24
        targetSdk 34
        versionCode 1
        versionName "1.0"
    }

    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
    
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_17
        targetCompatibility JavaVersion.VERSION_17
    }
    
    kotlinOptions {
        jvmTarget = '17'
    }
    
    // TFLiteファイルを圧縮しない
    aaptOptions {
        noCompress "tflite"
    }
}

dependencies {
    // TensorFlow Lite
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0'
    
    // CameraX
    def camerax_version = "1.3.0"
    implementation "androidx.camera:camera-core:${camerax_version}"
    implementation "androidx.camera:camera-camera2:${camerax_version}"
    implementation "androidx.camera:camera-lifecycle:${camerax_version}"
    implementation "androidx.camera:camera-view:${camerax_version}"
    
    // その他
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'com.google.android.material:material:1.10.0'
    implementation 'androidx.constraintlayout:constraintlayout:2.1.4'
    implementation 'androidx.lifecycle:lifecycle-runtime-ktx:2.6.2'
}
```

#### 3.2.2 プロジェクトレベルのbuild.gradle

`build.gradle` (Project level)を確認:

```gradle
buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath 'com.android.tools.build:gradle:8.1.0'
        classpath 'org.jetbrains.kotlin:kotlin-gradle-plugin:1.9.0'
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}
```

### 3.3 モデルファイル配置

#### 3.3.1 assetsディレクトリの作成

```
app/src/main/assets/
```

#### 3.3.2 モデルファイルのコピー

変換したTFLiteモデルを`app/src/main/assets/`にコピー:

```powershell
# プロジェクトルートから実行
Copy-Item models\tflite\i3d_int8.tflite app\src\main\assets\
```

#### 3.3.3 ラベルファイルの作成

`app/src/main/assets/kinetics_labels.txt`を作成:

Kinetics-400のラベルリストを取得（GitHubリポジトリから）し、以下の形式で保存:

```
0: abseiling
1: air_drumming
2: answering_questions
...
399: brushing_teeth
```

### 3.4 CameraX設定

#### 3.4.1 AndroidManifest.xml

`app/src/main/AndroidManifest.xml`に権限を追加:

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-feature android:name="android.hardware.camera" android:required="true" />
    
    <application
        ...>
        <activity ...>
            ...
        </activity>
    </application>
</manifest>
```

### 3.5 推論コード実装

#### 3.5.1 TFLiteModelLoader.kt

`app/src/main/java/com/example/toothbrushing/TFLiteModelLoader.kt`:

```kotlin
package com.example.toothbrushing

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteModelLoader(private val context: Context) {
    
    fun loadModel(modelName: String): Interpreter {
        val modelBuffer = loadModelFile(modelName)
        
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            setUseNNAPI(true)
        }
        
        return Interpreter(modelBuffer, options)
    }
    
    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val assetManager = context.assets
        val fileDescriptor = assetManager.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}
```

#### 3.5.2 ImageUtils.kt

`app/src/main/java/com/example/toothbrushing/utils/ImageUtils.kt`:

```kotlin
package com.example.toothbrushing.utils

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ImageUtils {
    
    fun prepareInputTensor(frames: List<Bitmap>): ByteBuffer {
        val inputShape = intArrayOf(1, frames.size, 224, 224, 3)
        val inputBuffer = ByteBuffer.allocateDirect(
            4 * inputShape[0] * inputShape[1] * inputShape[2] * 
            inputShape[3] * inputShape[4]
        )
        inputBuffer.order(ByteOrder.nativeOrder())
        
        for (frame in frames) {
            val resized = Bitmap.createScaledBitmap(frame, 224, 224, true)
            val pixels = IntArray(224 * 224)
            resized.getPixels(pixels, 0, 224, 0, 0, 224, 224)
            
            for (pixel in pixels) {
                // RGB値を[0,1]に正規化
                inputBuffer.putFloat((pixel shr 16 and 0xFF) / 255.0f)
                inputBuffer.putFloat((pixel shr 8 and 0xFF) / 255.0f)
                inputBuffer.putFloat((pixel and 0xFF) / 255.0f)
            }
        }
        
        inputBuffer.rewind()
        return inputBuffer
    }
    
    fun Bitmap.resizeToModelInput(): Bitmap {
        return Bitmap.createScaledBitmap(this, 224, 224, true)
    }
}
```

#### 3.5.3 LabelUtils.kt

`app/src/main/java/com/example/toothbrushing/utils/LabelUtils.kt`:

```kotlin
package com.example.toothbrushing.utils

import android.content.Context
import java.io.BufferedReader

object LabelUtils {
    
    fun loadLabels(context: Context, filename: String): List<String> {
        return context.assets.open(filename).bufferedReader().useLines { lines ->
            lines.map { it.substringAfter(": ").trim() }.toList()
        }
    }
    
    fun getBrushingTeethProbability(
        output: FloatArray,
        labels: List<String>
    ): Float {
        val brushingTeethIndex = labels.indexOf("brushing_teeth")
        return if (brushingTeethIndex >= 0 && brushingTeethIndex < output.size) {
            output[brushingTeethIndex]
        } else {
            0f
        }
    }
}
```

#### 3.5.4 ToothBrushingAnalyzer.kt

`app/src/main/java/com/example/toothbrushing/ToothBrushingAnalyzer.kt`:

```kotlin
package com.example.toothbrushing

import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import com.example.toothbrushing.utils.ImageUtils
import com.example.toothbrushing.utils.LabelUtils
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ToothBrushingAnalyzer(
    private val interpreter: Interpreter,
    private val labels: List<String>,
    private val onResult: (Float) -> Unit
) : ImageAnalysis.Analyzer {
    
    private val frameBuffer = mutableListOf<Bitmap>()
    private val bufferSize = 16
    private var isProcessing = false
    
    override fun analyze(imageProxy: ImageProxy) {
        if (isProcessing) {
            imageProxy.close()
            return
        }
        
        isProcessing = true
        
        val bitmap = imageProxyToBitmap(imageProxy)
        frameBuffer.add(bitmap)
        
        if (frameBuffer.size >= bufferSize) {
            processFrames(frameBuffer.toList())
            frameBuffer.clear()
        }
        
        imageProxy.close()
        isProcessing = false
    }
    
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer
        
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        
        // YUV → Bitmap変換（簡易実装、実際は適切な変換ライブラリを使用推奨）
        // ここでは簡略化のため、直接Bitmapに変換する処理を実装
        // 実際の実装では、YuvImageを使用することを推奨
        
        return Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
    }
    
    private fun processFrames(frames: List<Bitmap>) {
        val inputBuffer = ImageUtils.prepareInputTensor(frames)
        
        val outputShape = interpreter.getOutputTensor(0).shape()
        val outputBuffer = ByteBuffer.allocateDirect(
            4 * outputShape[0] * outputShape[1]
        )
        outputBuffer.order(ByteOrder.nativeOrder())
        
        interpreter.run(inputBuffer, outputBuffer)
        
        outputBuffer.rewind()
        val outputArray = FloatArray(outputShape[1])
        outputBuffer.asFloatBuffer().get(outputArray)
        
        val probability = LabelUtils.getBrushingTeethProbability(outputArray, labels)
        onResult(probability)
    }
}
```

### 3.6 UI統合

#### 3.6.1 activity_main.xml

`app/src/main/res/layout/activity_main.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent">
    
    <androidx.camera.view.PreviewView
        android:id="@+id/viewFinder"
        android:layout_width="match_parent"
        android:layout_height="match_parent" />
    
    <TextView
        android:id="@+id/scoreTextView"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="スコア: 0%"
        android:textSize="24sp"
        android:textColor="@android:color/white"
        android:background="#80000000"
        android:padding="16dp"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        android:layout_marginTop="32dp" />
    
</androidx.constraintlayout.widget.ConstraintLayout>
```

#### 3.6.2 MainActivity.kt

`app/src/main/java/com/example/toothbrushing/MainActivity.kt`:

```kotlin
package com.example.toothbrushing

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.toothbrushing.utils.LabelUtils
import org.tensorflow.lite.Interpreter
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    
    private lateinit var viewFinder: PreviewView
    private lateinit var scoreTextView: TextView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var interpreter: Interpreter
    private lateinit var labels: List<String>
    
    private val CAMERA_PERMISSION_REQUEST_CODE = 1001
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        viewFinder = findViewById(R.id.viewFinder)
        scoreTextView = findViewById(R.id.scoreTextView)
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        if (checkCameraPermission()) {
            setupCamera()
        } else {
            requestCameraPermission()
        }
    }
    
    private fun checkCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }
    
    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_REQUEST_CODE
        )
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            setupCamera()
        }
    }
    
    private fun setupCamera() {
        // モデルとラベルの読み込み
        val modelLoader = TFLiteModelLoader(this)
        interpreter = modelLoader.loadModel("i3d_int8.tflite")
        labels = LabelUtils.loadLabels(this, "kinetics_labels.txt")
        
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }
            
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
            
            val analyzer = ToothBrushingAnalyzer(interpreter, labels) { probability ->
                runOnUiThread {
                    updateScore(probability)
                }
            }
            
            imageAnalysis.setAnalyzer(cameraExecutor, analyzer)
            
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
            } catch (ex: Exception) {
                ex.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun updateScore(probability: Float) {
        val score = (probability * 100).toInt()
        scoreTextView.text = "歯磨きスコア: $score%"
        
        scoreTextView.setTextColor(
            when {
                score >= 70 -> ContextCompat.getColor(this, android.R.color.holo_green_dark)
                score >= 50 -> ContextCompat.getColor(this, android.R.color.holo_orange_dark)
                else -> ContextCompat.getColor(this, android.R.color.holo_red_dark)
            }
        )
    }
    
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        interpreter.close()
    }
}
```

## 4. テスト・検証

### 4.1 モデル変換の検証

#### 4.1.1 TFLiteモデルの検証

`scripts/verify_tflite_model.py`を作成:

```python
"""
TFLiteモデルを検証
"""
import tensorflow as tf
import numpy as np

def verify_tflite_model(model_path: str):
    """TFLiteモデルを検証"""
    
    # インタープリターを作成
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # 入力・出力の詳細を表示
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("入力詳細:")
    for detail in input_details:
        print(f"  名前: {detail['name']}")
        print(f"  形状: {detail['shape']}")
        print(f"  型: {detail['dtype']}")
    
    print("\n出力詳細:")
    for detail in output_details:
        print(f"  名前: {detail['name']}")
        print(f"  形状: {detail['shape']}")
        print(f"  型: {detail['dtype']}")
    
    # テスト推論を実行
    print("\nテスト推論を実行中...")
    test_input = np.random.rand(1, 16, 224, 224, 3).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"出力形状: {output.shape}")
    print(f"出力範囲: [{output.min():.4f}, {output.max():.4f}]")
    
    # "brushing_teeth"クラスのインデックスを確認（ラベルファイルから）
    # 通常は399（Kinetics-400の場合）
    if output.shape[1] >= 400:
        brushing_teeth_prob = output[0][399]
        print(f"\n'brushing_teeth'クラスの確率: {brushing_teeth_prob:.4f}")
    
    print("\n✓ モデル検証が完了しました")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        verify_tflite_model(sys.argv[1])
    else:
        verify_tflite_model("./models/tflite/i3d_int8.tflite")
```

実行:

```powershell
python scripts/verify_tflite_model.py models\tflite\i3d_int8.tflite
```

### 4.2 Androidアプリでの動作確認手順

#### 4.2.1 ビルドと実行

1. Android Studioでプロジェクトを開く
2. `Build → Make Project`でビルド
3. 実機またはエミュレーターを接続
4. `Run → Run 'app'`で実行

#### 4.2.2 動作確認チェックリスト

- [ ] カメラが正常に起動する
- [ ] カメラプレビューが表示される
- [ ] スコア表示が更新される
- [ ] 歯磨き動作でスコアが上昇する
- [ ] アプリがクラッシュしない
- [ ] メモリリークがない（長時間実行テスト）

#### 4.2.3 ログ確認

Android StudioのLogcatで以下を確認:

```
# モデル読み込み
D/TFLiteModelLoader: Model loaded successfully

# 推論実行
D/ToothBrushingAnalyzer: Inference completed, probability: 0.XX

# エラー
E/ToothBrushingAnalyzer: Error: ...
```

### 4.3 パフォーマンス測定方法

#### 4.3.1 推論時間の測定

`ToothBrushingAnalyzer.kt`に計測コードを追加:

```kotlin
private fun processFrames(frames: List<Bitmap>) {
    val startTime = System.currentTimeMillis()
    
    val inputBuffer = ImageUtils.prepareInputTensor(frames)
    
    // ... 推論実行 ...
    
    val endTime = System.currentTimeMillis()
    val inferenceTime = endTime - startTime
    android.util.Log.d("Performance", "Inference time: ${inferenceTime}ms")
    
    val probability = LabelUtils.getBrushingTeethProbability(outputArray, labels)
    onResult(probability)
}
```

#### 4.3.2 メモリ使用量の確認

Android StudioのProfilerを使用:

1. `View → Tool Windows → Profiler`
2. アプリを実行
3. Memoryタブでメモリ使用量を確認
4. リークがないか確認

#### 4.3.3 バッテリー消費の確認

1. デバイスの設定 → バッテリー → アプリの使用状況
2. 長時間実行テスト（30分以上）
3. バッテリー消費率を確認

## 5. 次のステップ

実装が完了したら、以下の改善を検討してください:

1. **ファインチューニング**: 子供の歯磨きデータでモデルを再訓練
2. **UI改善**: より魅力的なフィードバックUI
3. **音声フィードバック**: 音声ガイダンスの追加
4. **データ記録**: スコアの履歴保存
5. **ゲーミフィケーション**: アチーブメントシステムの追加

詳細は`troubleshooting.md`と`references.md`を参照してください。

