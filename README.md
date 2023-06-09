# みかんの異常検知の試作

## タスク
- 正常画像のみを用いた異常検知手法の試作
- 判断根拠を示すために異常箇所を可視化

## 手法
[Improving Unsupervised Defect Segmentation by Applying Structural Similarity To Autoencoders](https://arxiv.org/abs/1807.02011)を参考に実装
- AutoEncoder + SSIM Lossを用いた教師なし学習
- 正常画像のみを用いて、画像を再構成するAutoEncoderを学習

## データセット
自作のみかんデータセット(RGB x 256 x 256, 正常画像: 190枚 異常画像: 30枚)
- 上面、側面、底面を撮影し、みかん領域を手動クロッピング + パディング

## 学習
- Optimaizer : Adam
- Augmentation: Random Crop, Horizontal Flip
- Loss: SSIM

## 結果(定性)

<img src="https://github.com/kotaro-kinoshita/orange_anomaly_detection/blob/main/demo/demo1.jpg"> <img src="https://github.com/kotaro-kinoshita/orange_anomaly_detection/blob/main/demo/demo2.jpg"> <img src="https://github.com/kotaro-kinoshita/orange_anomaly_detection/blob/main/demo/demo3.jpg"> 

<img src="https://github.com/kotaro-kinoshita/orange_anomaly_detection/blob/main/demo/heatmap1.jpg"> <img src="https://github.com/kotaro-kinoshita/orange_anomaly_detection/blob/main/demo/heatmap2.jpg"> <img src="https://github.com/kotaro-kinoshita/orange_anomaly_detection/blob/main/demo/heatmap3.jpg">






