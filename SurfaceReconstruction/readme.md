# 任务和文件说明

实现了基于MLP的从点云重建三维物体表面
HW4 是基础任务的代码，linux环境运行sh run1.sh 即可依次开始训练并重建物体。
HW4_fourier 是在基础任务代码上修改的，linux环境运行sh run2.sh 即可依次开始训练并重建物体。
Reuslts是训练的结果，results1中是基础任务的结果，results2中是拓展任务的结果。

两个任务运行后生成的三维物体都以.ply格式储存在./results文件夹下 checkpoints储存在./checkpoints文件夹下
训练完成后想要加载模型生成数据，在各自目录运行test.py --data_path=xxx --obj_index=x，即可加载
checkpoints测试，生成结果也在./results文件夹下，其中data_path是数据文件路径，obj_index是数据编号，按data中的顺序1——5。
