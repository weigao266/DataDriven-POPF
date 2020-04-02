#!/bin/bash
#将一个目录下的一些文件移动到另一个目录下

dir_figures="/Users/swg/Desktop/PLF/figures"  #可修改绝对路径；

#mv ./*.eps /Users/swg/Weyun/iDirIMM/iDirIMM-tex.  #移动文件
cp -i $dir_figures/* /Users/swg/Weyun/iDirIMM/iDirIMM-tex   #复制文件
