###���д�����ܳ��ֵ����⣺
---
 (1) pydotplus.graphviz.InvocationException: GraphViz's executables not found

  	û�а�װGraphViz��s executables.������pip��װ��Graphviz,����Graphviz����һ��python tool������Ȼ��Ҫ��װGraphViz��s executables.
  	��GraphViz��װĿ¼��binĿ¼�ŵ�����������path·����
  	ubuntu 14.04     sudo apt-get install graphviz
  	windows�£�windows�汾���ز���װ����ַ��http://www.graphviz.org/download/
		python��ִ�У�
		import os
		os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'