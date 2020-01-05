# HAN-readability-classification

### 这是[唐玉玲]的HAN模型部分的试验代码。包含Chinese-readability-classification和English-readability-classification
#### 由于中文的数据处理方式的差别，这个项目中包含两个子项目，一个是英文可读性分级，另一个是中文可读性分级。其中英文可读性分级的数据集是ose_data和wbt_data。
### 由于中文数据集涉及我所属实验室的数据隐私，不便于上传公开，所以就不能在这里上传公开。

### 代码运行步骤：
#### 1. 每个子项目里都有四个Python文件，分别是：1) data_processor.py  2) attention.py    3) model.py    4) train.py
#### 2. 直接运行train.py就可以。  python train.py   

### 在English-readability-classification中，
#### ose_data是OneStopEnglish数据集分好训练，验证，测试集之后的文件夹，数据格式为 index，level，content。其中content还保留着段落的格式信ose_data_2是丢掉了index一列，只保留level和content，其中content的内容去掉了段落格式，全都变成一个段落。
#### wbt_data是WeeBit数据集切分好训练，验证，测试集之后的文件夹，数据格式为 index, level, content。其中content还保留着段落的格式信息。wbt_data_2是丢掉了index一列，只保留level和content，其中content的内容去掉了段落格式，全部变成一个段落。
#### 在我的HAN模型中，使用的是ose_data 和 wbt-data ,代码中的输入是ose_data的数据，若要验证wbt数据集，只需要在data_processor.py中将数据文件改成wbt的数据文件即可。
