[English](README.md) | **日本語**

# ALICE-Video

A.L.I.C.E. エコシステム向けビデオコーデック・処理ライブラリ。ビデオパイプラインの中核コンポーネントを純Rustで実装。

## 機能

- **ピクセルフォーマット** — YUV 4:2:0プレーナーとRGBインターリーブの双方向変換
- **フレームタイプ** — I/P/Bフレーム分類、GOP構造
- **動き補償** — ブロックマッチング、動きベクトル推定・予測
- **DCT・量子化** — 8x8離散コサイン変換、設定可能な量子化テーブル
- **エントロピー符号化** — 可変長符号化による圧縮ビットストリーム出力
- **解像度スケーリング** — バイリニア・最近傍リサンプリング
- **ビットレート制御** — CBR/VBRレート制御、品質ターゲティング
- **コンテナフォーマット** — 基本的なMux/Demuxサポート

## アーキテクチャ

```
生フレーム (RGB/YUV)
  │
  ├── pixel       — フォーマット変換 (YUV↔RGB)
  ├── frame       — I/P/Bフレーム分類、GOP
  ├── motion      — ブロックマッチング、動きベクトル
  ├── dct         — DCT変換、量子化
  ├── entropy     — 可変長符号化
  ├── scale       — 解像度リサンプリング
  ├── bitrate     — レート制御 (CBR/VBR)
  └── container   — Mux/Demux
```

## 使用例

```rust
use alice_video::{RgbPixel, YuvPixel};

let rgb = RgbPixel::new(255, 128, 64);
let yuv = rgb.to_yuv();
let back = yuv.to_rgb();
```

## ライセンス

MIT OR Apache-2.0
