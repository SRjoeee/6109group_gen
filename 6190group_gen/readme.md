### 安装环境
跑metrics需要一两个额外的包
pip install git+https://github.com/openai/CLIP.git
pip install lpips

如果还有别的包Missing可以问我或者GPT


### 数据集
数据集是一个深度图生成的数据集，一共有10个class，每个三张图，每张图5个prompt
prompt 在 data/imnetr-ti2i.yaml里，需要根据每张输入图调用对应的5个prompt，对每个prompt生成一次
run_sdedit是写好调用逻辑的版本，可以对照着改controlnet那个
可能需要检查一下这些文件调用逻辑以及数据集的folder path，因为是受文件夹结构影响的

### Controlnet
那个conditioning scale就是controlnet的强度scale
guidance scale是sd模型里控制prompt对结果影响强度的

### Sdedit
那个strength就是最重要的那个parameter strength.


### Metrics
在eval.py里
需要三个folder，一个是生成好的，一个是输入的生成图，一个是ground truth (data/GT),要改一下eval.py里的path设置
解释一下因为原数据集是图像转 深度图做出来的，所以有GT图像
metrics是三个：
self-similarity distance, 比较**GT图像** 和 **生成图像**  越低说明结构越相近，越低越好
LPIPS, 比较**输入深度图** 和 **生成图像**  越高说明生成图在texture上越不像深度图，就是越像好的图片，越高越好
CLIP score, 比较**prompt** 和 **生成图像**  越高说明生成图和prompt越符合，越高越好






