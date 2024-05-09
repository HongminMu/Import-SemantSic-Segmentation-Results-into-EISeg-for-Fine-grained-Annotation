# Import-Semantic-Segmentation-Results-into-EISeg-for-Fine-grained-Annotation
Directly annotating an image from scratch is very time-consuming, but using existing models to generate coarse labels and then refining them can greatly save time.
效果预览：
![image](https://github.com/HongminMu/Import-Semantic-Segmentation-Results-into-EISeg-for-Fine-grained-Annotation/assets/57067148/5da80cf9-fbbb-4487-8e70-5d3fc1a2beb1)

实现步骤：
1将输入图像：![GP010364_frame_000105_rgb_anon](https://github.com/HongminMu/Import-Semantic-Segmentation-Results-into-EISeg-for-Fine-grained-Annotation/assets/57067148/8dbf0ba9-d19f-487a-8e0b-ba89618ee648)
使用基准网络处理，得到：![GP010364_frame_000105_rgb_anon](https://github.com/HongminMu/Import-Semantic-Segmentation-Results-into-EISeg-for-Fine-grained-Annotation/assets/57067148/e4f4b755-b0c8-4940-b0cd-74d9428de4fd) （这一步是为了让你会用predict.py）

2将predict_with_jsons_cityscapes.py放在D:\PaddleSeg-release-2.8\predict_withjson.py
以与predict.py相同的配置参数，运行py文件，输出一个annotations.json：
![image](https://github.com/HongminMu/Import-Semantic-Segmentation-Results-into-EISeg-for-Fine-grained-Annotation/assets/57067148/c8a1bbd4-932d-4a57-a1ba-255115609372)

3同文件夹中新建lable文件夹，将annotations.json放入该文件夹
![image](https://github.com/HongminMu/Import-Semantic-Segmentation-Results-into-EISeg-for-Fine-grained-Annotation/assets/57067148/41383806-9a38-4edb-9cce-151caf2ddbc0)

4用EISeg打开该文件夹：
![image](https://github.com/HongminMu/Import-Semantic-Segmentation-Results-into-EISeg-for-Fine-grained-Annotation/assets/57067148/5da80cf9-fbbb-4487-8e70-5d3fc1a2beb1)

就可以进行再标注啦！
