# Dentifrice Support System

子供向け歯磨きアプリケーション向けの、リアルタイム歯磨き動作認識システムの設計ドキュメントと実装ガイドです。

## 概要

本プロジェクトは、既存の大規模データセットで学習済みの認識モデル（I3D from Kinetics、NTU RGB+Dベース）をTensorFlow Lite経由でAndroidアプリに統合し、リアルタイムで歯磨き動作を認識・評価するための包括的な設計ドキュメントと実装ガイドを提供します。

## 主な特徴

- **既存モデルの活用**: 追加のデータ収集や学習コストを最小限に抑えながら、高精度な動作認識を実現
- **モバイル最適化**: TensorFlow Liteによる軽量化・量子化で、モバイルデバイスでのリアルタイム推論を実現
- **完全無料**: すべてオープンソースツールのみを使用（TensorFlow Hub、TensorFlow Lite、Android Studio）
- **プライバシー保護**: すべての処理がデバイス上で実行され、データが外部に送信されない

## ドキュメント

### 📘 [設計ドキュメント](docs/design_document.md)
システム全体の設計、モデル選択、アーキテクチャ、変換設計、Android統合設計を詳しく説明しています。

### 📖 [実装ガイド](docs/implementation_guide.md)
環境構築からモデル変換、Android統合、テストまで、ステップバイステップの実装手順を提供しています。

### 🔧 [トラブルシューティング](docs/troubleshooting.md)
よくある問題とその解決方法をまとめています。

### 📚 [参考資料](docs/references.md)
TensorFlow、Android開発、データセット、GitHubリポジトリなどの有用なリソースをまとめています。

## クイックスタート

1. **環境構築**
   - Python環境（uv使用）のセットアップ
   - Android Studioのインストール
   - 詳細は[実装ガイド - 環境構築](docs/implementation_guide.md#1-環境構築)を参照

2. **モデル変換**
   - TensorFlow HubからI3Dモデルを取得
   - TensorFlow Lite形式に変換
   - 詳細は[実装ガイド - モデル変換](docs/implementation_guide.md#2-モデル変換ステップ)を参照

3. **Android統合**
   - Androidプロジェクトのセットアップ
   - CameraXとTensorFlow Liteの統合
   - 詳細は[実装ガイド - Android統合](docs/implementation_guide.md#3-android統合ステップ)を参照

## 技術スタック

- **モデルフレームワーク**: TensorFlow Lite 2.14.0
- **変換ツール**: TensorFlow Hub, ONNX Runtime
- **Android開発**: CameraX, TensorFlow Lite Support Library
- **開発環境**: Python (uv管理), Android Studio
- **対象OS**: Android 15以降

## プロジェクト構造

```
dentifrice_support_system/
├── docs/
│   ├── design_document.md          # メイン設計ドキュメント
│   ├── implementation_guide.md     # 実装ガイド
│   ├── troubleshooting.md          # トラブルシューティング
│   └── references.md               # 参考資料
├── models/                          # モデルファイル（.gitignore対象）
│   ├── saved_model/
│   └── tflite/
├── scripts/                         # 変換スクリプト（将来追加予定）
├── .gitignore
└── README.md
```

## 対応モデル

### I3D (Inflated 3D ConvNet) from Kinetics
- **データセット**: Kinetics-400/600
- **認識クラス**: "brushing teeth"
- **モデルサイズ**: 約75-125MB（int8量子化後）
- **推論速度**: 約30-50ms/クリップ（NNAPI使用時）

### NTU RGB+Dベースモデル
- **データセット**: NTU RGB+D 60/120
- **認識クラス**: "brush teeth"
- **モデルサイズ**: 約50-100MB（軽量版）
- **推論速度**: 高速（軽量アーキテクチャ）

詳細な比較は[設計ドキュメント - モデル選択と比較](docs/design_document.md#2-モデル選択と比較)を参照してください。

## ライセンス

本プロジェクトのドキュメントは、教育・研究目的で自由に使用できます。

使用するモデル（I3D、NTU RGB+D）のライセンスについては、それぞれの公式リポジトリを確認してください。

## 貢献

バグ報告、機能要望、ドキュメントの改善提案は、Issueトラッカーでお知らせください。

## 参考リンク

- [TensorFlow Lite公式サイト](https://www.tensorflow.org/lite)
- [TensorFlow Hub - I3Dモデル](https://tfhub.dev/deepmind/i3d-kinetics-400/1)
- [Android CameraX](https://developer.android.com/training/camerax)
- [詳細な参考資料一覧](docs/references.md)

## 更新履歴

- **2024-12-15**: 初版作成
  - 設計ドキュメント作成
  - 実装ガイド作成
  - トラブルシューティングガイド作成
  - 参考資料ドキュメント作成