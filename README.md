# Import-Semantic-Segmentation-Results-into-EISeg-for-Fine-grained-Annotation
Directly annotating an image from scratch is very time-consuming, but using existing models to generate coarse labels and then refining them can greatly save time.

输入图像：![GP010364_frame_000105_rgb_anon](https://github.com/HongminMu/Import-Semantic-Segmentation-Results-into-EISeg-for-Fine-grained-Annotation/assets/57067148/8dbf0ba9-d19f-487a-8e0b-ba89618ee648)

使用基准网络处理，得到：![GP010364_frame_000105_rgb_anon](https://github.com/HongminMu/Import-Semantic-Segmentation-Results-into-EISeg-for-Fine-grained-Annotation/assets/57067148/e4f4b755-b0c8-4940-b0cd-74d9428de4fd)

再导入EISeg后：
![1715250762169](https://github.com/HongminMu/Import-Semantic-Segmentation-Results-into-EISeg-for-Fine-grained-Annotation/assets/57067148/ebfd94d4-bce5-4577-9076-7984980dabcc)

之后就可以进行再标注啦！
