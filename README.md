# DATA620004 神经网络与深度学习 期末作业 第三部分

**陈子健 22110980002**

这是我们小组期末作业的第三部分。本部分使用[FFmpeg](https://github.com/FFmpeg/FFmpeg)+[COLMAP](https://github.com/colmap/colmap)+[Instant-NGP](https://github.com/NVlabs/instant-ngp)对复旦校园中的部分雕塑进行了3D建模。我们期末作业的另外两个部分分别见[第一题]()和[第二题]()。本部分的报告中有视频需要展示，请直接点击连接中的[markdown文件](./report.md)。

## 网盘地址

- [Instant-NGP(包括已训练好的模型)](https://pan.baidu.com/s/1EnPhPK0G0YvNiNfH8OCGLA?pwd=x8ew)
- [Requirements](https://pan.baidu.com/s/1QoyVv9NJitTBoE42pR-jBw?pwd=mq26)

## Requirements

### 环境

**说明：** 本部分的环境比较特殊，由于Instant-NGP的原始文件是用CUDA写的，然后使用CMAKE来编译。我们是在Windows中完成的上述操作，所以获得的`instant-ngp.exe`也只能在Windows中跑通。我们作业所使用的环境(系租用[矩池云](https://matgo.cn)上的服务器)为：
- Windows Server 2019
- Python 3.8
- Pytorch 1.12
- CUDA 11.3
- cuDNN 8.2.0

### 安装

请事先安装好以下程序

- [COLMAP](https://demuc.de/colmap/#download)(必须)
- [FFmpeg](https://ffmpeg.org/download.html)(必须)
- [VLC Media Player](https://www.videolan.org/vlc/)(建议)
- [Infranview](https://www.irfanview.com/)(建议)

也可以从我的[网盘](https://pan.baidu.com/s/1QoyVv9NJitTBoE42pR-jBw?pwd=mq26)上直接下载`Downloads.7z`文件，直接解压即可，不用安装。安装完成后请**务必**将`colmap.exe`和`ffmpeg.exe`所在的文件夹手动添加到系统的PATH中。其中COLMAP安装后还需将安装目录下的`lib`文件夹内的**所有文件**(不是文件夹)复制到`bin`下。


本repo中的`./Instant-NGP-for-RTX-2000/scripts/colmap2nerf.py`中虽然设置了没安装好COLMAP和FFmpeg时自动从网上下载安装的语句，但一是网络连接不稳定且速度较慢，二是无法自动将COLMAP和FFmpeg的写入系统的PATH。

最后请从[原始的repo](https://github.com/NVlabs/instant-ngp)中安装对应版本的Instant-NGP或直接从[网盘](https://pan.baidu.com/s/1EnPhPK0G0YvNiNfH8OCGLA?pwd=x8ew)中下载已包括我生成好的一些模型的压缩包并解压。

### Python包
- `commentjson`
- `imageio`
- `numpy`
- `opencv-python-headless`
- `pybind11`
- `pyquaternion`
- `scipy`
- `tqdm`

通过`requirements.txt`安装
```
cd [your-own-path]\Instant-NGP-for-RTX-2000
pip install -r requirements.txt
```

### 运行过程

#### 只看结果

```
cd [your-own-path]\Instant-NGP-for-RTX-2000
instant-ngp subuqing
```
上面的`subuqing`可以换成`chenwangdao`、`sijiao`、`lvbeishisi`。

#### 重新训练

马相伯像是没有训练的。下面代码可以对其进行训练。

- **第一步：** 用FFmpeg把mp4格式输入视频的按一秒两帧拆分成图片

```
cd [your-own-path]\Instant-NGP-for-RTX-2000
python scripts\colmap2nerf.py --video_in maxiangbo\maxiangbo.mp4 --video_fps 2 --run_colmap --aabb_scale 32 --overwrite
```
- **第二步：** 图片生成后放在对应视频所在文件夹的`images`文件夹下，原作者提示进行下一步前务必先删除模糊以及与需要建模的物体无关的帧。
- **第三步：** 用COLMAP生成数据集(matching)，COLMAP会自动计算场景的中心坐标和x轴、y轴和z轴正负方向延展的最大坐标。
```
cd maxiangbo
python [your-own-path]\Instant-NGP-for-RTX-2000\scripts\colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 32 --overwrite
```
- **第四步：** 查看结果
```
cd ..
instant-ngp maxiangbo
```

### 结果展示

这里仅展示一个demo，为子彬院前的苏步青像。

