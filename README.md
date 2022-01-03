# 基于PCISPH方法的简易固液耦合

## 作业来源

> 通过一个简易2D场景——小球掉入水中，来复现论文：
>
> 1. **Versatile Rigid-Fluid Coupling for Incompressible SPH** (TOG 2012)
> 2. **Predictive-Corrective Incompressible SPH** (TOG 2009)

## 运行方式

#### 运行环境：

> [Taichi] version 0.8.8, llvm 10.0.0, commit 7bae9c77, win, python 3.7.9

#### 运行：

>  python PCISPH.py

## 效果展示
> ![](data\video.gif)

## 整体结构

```
-LICENSE
-|data
	-|video.gif
-README.MD
-PCISPH.py
```

## 实现细节：

> 参考了ti example pbf2d中的particle2grid部分
>
> 边界处理也比较简单，强制不出边界置垂直于边界的速度分量为0，固体做了简易碰撞的反弹
>
> 其余均参照论文中的算法流程进行复现

![](data\PCISPH.png)

![](data\Rigid-Fluid.png)

