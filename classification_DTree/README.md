###运行代码可能出现的问题：
---
 (1) pydotplus.graphviz.InvocationException: GraphViz's executables not found

  	没有安装GraphViz‘s executables.我是用pip安装的Graphviz,但是Graphviz不是一个python tool，你仍然需要安装GraphViz‘s executables.
  	将GraphViz安装目录的bin目录放到环境变量的path路径中
  	ubuntu 14.04     sudo apt-get install graphviz
  	windows下：windows版本下载并安装，地址：http://www.graphviz.org/download/
		python下执行：
		import os
		os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'