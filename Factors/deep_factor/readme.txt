x：
	当下时刻高频因子回看一定时间的特征(采样或不采样)
y：
	当下时刻一定周期的收益率(e.g. 30min, 240min)

我们对训练集的数据(240min)进行一定频率采样，并按/home/largefile/zjwang/filter_data的数据进行过滤，分为SH/SZ分别训练。
滚动对每个给定时刻回看一定时间特征，预测一定周期的收益率
如果预测30min，则去掉开头5min和最后29min
如果预测240min，则去掉开头5min，结尾4min



###==================================================================

(**表示代码还没实现，*表示还没debug)mlp只在test00试验，其他全部选择lstm，mlp模式的code没有相应更新
*test：对池子股票使用VAE
test_00: base
test_06：在test_00基础上加入k线因子（主要在Exp.py的get_data_mode有区别）
test_07: 加入RevIN (在model.py有区别)
test_08：筛选部分因子(input) （去掉一些因子，主要在Exp.py的get_data_mode有区别）
test_10：单个dataloader（之前分成多个dataloder了） 可以全局shuffle 并加入是否有k线选项。
test_11：在test_10基础上加入RevIN（在model.py有区别）
test_12：在test_10基础上单独对k线原始数据denorm（在model.py和Exp.py有区别，data需要分成两部分了）
test_13：对短期不采样，远的采用采样，在一个序列 (rolling函数改变)
test_14：对短期不采样，远的采用采样，分别两个序列，训练后concat （在13的基础上，model.py变化，默认短期(short_interval)为30，新增short_interval变量）

/home/largefile/public/zjwang/nn_factor/ic.ipynb   看ic的分布
###==================================================================

以base的test_10为例。
运行指令：
        nohup sh ./run.sh $1 $2 $3 $4 $5 > $6.txt 2>&1 & 
        比如：nohup sh ./run.sh lstm SH 1 nol1 1 > lstm_SH_1.txt 2>&1 &
运行指令说明：
        $1：模型名字，有 lstm和mlp两个选项；
        $2：筛选的股票，有SH和SZ两个选项；
        $3：试验的序号
        $4：是否loss加入l1正则化。l1为加入，其他为不加入
        $5：使用的gpu序号。0或1
文件说明：
    main.py 运行的主要文件，包含一些参数的调整；
    Exp.py 试验的主要文件，包括获取数据逻辑、训练逻辑、验证逻辑等等
	     主要函数是数据处理的函数，get_data_mode函数
	          
    loss.py 损失函数
    model.py 模型
    dataset.py 数据集
    xxx.txt 实验的中间结果输出
    results 实验的最终结果。包括scaler(归一化)，model(模型)，ic.csv(每日不同时刻的ic，用np保存，建议用np读取)，date.csv（深度学习因子输出）
main.py 参数说明：
    params：训练的一些参数
        date_len：实验所用的日期，包括训练日期和验证日期。按目前已有的ytrue的最近日期除去10天开始算（10天本来打算当作测试集日期）
        back_days：当前时刻的特征所回看的日期
        interval：每天股票采样的频率（从第5分钟开始，到第210min为止）.如果一天一个因子，interval设置为0
        idx：在interval=0时，即一天一个因子的时候有效，表示训练哪个时刻的，一天240min，从0开始计数
        train_data_frac：训练集和测试集日期划分比例
        back_interval：回看多少分钟的特征
        interval_feature：回看的特征采样的频率
        model_name：使用的模型名字，lstm 或 mlp。
        lr：学习率
        regular：loss是否采用l1正则化。在linux运行指令设置
        alpha：若采用l1正则化，正则化的系数。
        device：模型训练使用的gpu。在linux运行指令设置
        dir_x：input地址
        dir_y：y_true地址
        dir_f：过滤股票数据地址
        dir_y_p：实验结果保存地址
        is_filter：是否过滤股票。默认过滤，不需要设置。
        ins：使用的是SZ还是SH。在linux运行指令设置
        epoches：训练的epoches
        freeze：训练时是否冻结除了最后一层线性层外所有的参数并使用ic作为loss
        lr_decay：在转换loss为ic的时候是否降低学习率。在freeze=True才有用。
        trans_epoch：在第几个epoch冻结并转换loss。在freeze=True才有用。
        single_num：单个dataloader保存的训练日期长度。在test_10合并dataloader后该参数没用。之前是报错oom，现在已修复。
        print_mode：是否输出一些中间过程。之前debug的时候用，设置为False即可。
        print_num：每多少个batch输出一次验证集结果。
        y_col_name：y_true所在的列名。
        batch_size：训练集每个batch的大小
        exp_idx：试验的序号。主要方便保存输出结果。在linux运行指令设置
        k：是否使用k线因子

    params_net：网络的一些参数
        mlp时：
            dims：中间层的一些维度。
            output_dim：最终输出的维度。之前是因为试验预测整天240min的y_true。设置为1即可
        lstm时：
            lstm_dim：lstm隐藏层的维度
            lstm_num：使用几层lstm。默认是使用lstm，也可以使用gru，没试验过，加个参数gru=True即可。
            output_dim：最终输出的维度
            nhead：transformer层的多头注意力机制的头数。主要要被input_dim整除，需要手动设置
            attention：是否使用transformer encoder。