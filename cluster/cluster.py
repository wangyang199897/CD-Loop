def run(chromosome_list, output_directory):

    infile1 = open('../predict/classification/output/output.bedpe', 'r')
    infile2 = open(output_directory + '/deleteSubMatrix/predict_chr' + str(chromosome_list[0]) + '_finall.txt', 'r')
    outfile = open(output_directory + 'beforeClustering/beforeClustering.bedpe', 'w')

    n = 0
    index_1 = []
    true = []
    next(infile1)
    for line in infile1:
        lines = line.strip('\n').split('\t')
        if lines[0] == '1':
            index_1.append(n)
            true.append(lines)
            n += 1
        else:
            n += 1
    data_list = []
    for line in infile2:
        lines = line.strip('\n').split('\t')
        data_list.append(lines)
    result = [data_list[i] for i in index_1]
    for i in range(len(index_1)):
        value_str = true[i][2]
        value_str = value_str.strip('[]')
        if value_str:
            values = value_str.split(',')
            if len(values) > 1:
                values_score = float(values[1])
            else:
                values_score = 0.0
        else:
            values_score = 0.0
        outfile.write(str(chromosome_list[0]) + '\t' + str(int((int(result[i][0]) - 1) * 5000)) + '\t' + str(
            int(int(result[i][0]) * 5000)) + '\t' + str(chromosome_list[0]) + '\t' + str(
            int((int(result[i][1]) - 1) * 5000)) + '\t' + str(int(int(result[i][1]) * 5000)) + '\t' +
                      true[i][0] + '\t' + true[i][1] + '\t' + str(values_score) + '\t' + true[i][3] + '\t' +
                      true[i][4] + '\t' + true[i][5] + '\n')

    # import click
    from sklearn.neighbors import KDTree
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    from matplotlib import pylab as plt
    from scipy.sparse import coo_matrix
    import pandas as pd

    def rhoDelta(data, resol, dc):
        pos = data[[1, 4]].to_numpy() // resol
        # remove singleton
        # 使用归一化后的位置数据构建KD树，以便进行最近邻搜索。leaf_size参数指定了叶子节点的大小，metric参数指定了距离度量的方法
        # # 开始
        posTree = KDTree(pos, leaf_size=30, metric='chebyshev')
        # 使用KD树查询在距离2单位范围内的最近邻点的索引和距离
        # NNindexes按照顺序存放每个点的半径范围内的点 是二维的  里边的每个列表存放的是第i个位置邻近点的索引值
        NNindexes, NNdists = posTree.query_radius(pos, r=2, return_distance=True)
        _l = []  # 创建一个空列表用于存储每个点的最近邻点数量
        for v in NNindexes:  # 对每个最近邻点的索引列表进行迭代
            _l.append(len(v))
        _l = np.asarray(_l)  # 将列表_l转换为NumPy数组
        # data = data[_l > 9].reset_index(drop=True) # 根据最近邻点的数量大于5的条件筛选数据，并重置索引。这一步的目的是去除掉孤立的
        data = data[_l > 15].reset_index(drop=True)  # l > 16 根据最近邻点的数量大于5的条件筛选数据，并重置索引。这一步的目的是去除掉孤立的
        pos = data[[1, 4]].to_numpy() // resol
        # val = data[6].to_numpy()
        # # 结束
        val = data[8].to_numpy()

        # 这段代码使用了NearestNeighbors类来计算密度
        # 计算每个点的密度
        X = np.array(pos)
        # 创建 NearestNeighbors 对象
        # k = 25 # 1 : k=25  0.1 : k=10
        k = 25  # 1 : k=25  0.1 : k=10  中间的测序深度 k = 15
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)  # 加1是为了排除自身
        # 计算密度
        distances, indices = nbrs.kneighbors(X)
        densities = 1.0 / (distances[:, k] + 0.1)

        rhos = []
        for i in range(len(densities)):
            rhos.append(densities[i])
        rhos = np.asarray(rhos)  # rhos存储的是去除掉孤立点后，对每个候选点计算密度
        posTree = KDTree(pos, leaf_size=30, metric='chebyshev')

        # calculate delta_i, i.e. distance to nearest point with larger rho
        _r = 100
        # 当sort_results=True时，query_radius函数返回的最近邻点的索引和距离将按照距离从小到大的顺序进行排序
        _indexes, _dists = posTree.query_radius(pos, r=_r, return_distance=True, sort_results=True)
        deltas = rhos * 0  # 创建一个与rhos数组相同大小的全零数组，用于存储每个数据点的delta值
        LargerNei = rhos * 0 - 1  # 创建一个与rhos数组相同大小的全零数组，并将其中的所有元素减1，用于存储每个数据点的最近邻点的索引

        # LargerNei存放的是一个点邻近的所有点中，比他的密度大的并且离他最近点的点的索引值
        # deltas存放的一个点邻近的所有点中，密度比他大，并且是离他最近的点的距离
        for i in range(len(_indexes)):
            idx = np.argwhere(rhos[_indexes[i]] > rhos[_indexes[i][0]])  # 找到rho值比当前数据点更大的最近邻点的索引
            # rhos[_indexes[i][0]]存储的是第一个值也是当前点
            if idx.shape[0] == 0:  # 如果没有找到一个比当前点大的密度的点
                deltas[i] = _dists[i][-1] + 1  # 距离设置成比当前邻居节点更大的基因组距离
            else:
                LargerNei[i] = _indexes[i][idx[0]]  # 索引值
                deltas[i] = _dists[i][idx[0]]  # 距离
        failed = np.argwhere(LargerNei == -1).flatten()
        while len(failed) > 1 and _r < 10000:
            _r = _r * 10
            _indexes, _dists = posTree.query_radius(pos[failed], r=_r, return_distance=True, sort_results=True)
            for i in range(len(_indexes)):
                idx = np.argwhere(rhos[_indexes[i]] > rhos[_indexes[i][0]])
                if idx.shape[0] == 0:
                    deltas[failed[i]] = _dists[i][-1] + 1
                else:
                    LargerNei[failed[i]] = _indexes[i][idx[0]]
                    deltas[failed[i]] = _dists[i][idx[0]]
            failed = np.argwhere(LargerNei == -1).flatten()

        data['rhos'] = rhos
        data['deltas'] = deltas

        return data

    def loopPDP(file_p):
        infile = open(file_p, 'r')
        cp_all = []
        for line in infile:
            line_temp = line.strip('\n').split('\t')
            bin1 = int(int(line_temp[2]) / 5000)
            bin2 = int(int(line_temp[5]) / 5000)
            if abs(bin1 - bin2) < 100:
                cp_all.append(line)
        infile.close()
        outfile = open(file_p, 'w')
        for cp in cp_all:
            outfile.write(cp)
        outfile.close()


    def pool(dc, candidates, resol, mindelta, minscore, output, refine):
        data = pd.read_csv(candidates, sep='\t', header=None)
        data = data[data[8] > minscore].reset_index(drop=True)
        data[['rhos', 'deltas']] = 0

        data = rhoDelta(data, resol=resol, dc=dc).reset_index(drop=True)
        targetData = data



        loopPds = []
        for chrom in set(targetData[0]):
            data = targetData[targetData[0] == chrom].reset_index(drop=True)

            pos = data[[1, 4]].to_numpy() // resol
            posTree = KDTree(pos, leaf_size=30, metric='chebyshev')

            rhos = data['rhos'].to_numpy()
            deltas = data['deltas'].to_numpy()
            centroid = np.argwhere(deltas > mindelta).flatten()

            _r = 100
            _indexes, _dists = posTree.query_radius(pos, r=_r, return_distance=True, sort_results=True)
            LargerNei = rhos * 0 - 1
            for i in range(len(_indexes)):
                idx = np.argwhere(rhos[_indexes[i]] > rhos[_indexes[i][0]])
                if idx.shape[0] == 0:
                    pass
                else:
                    LargerNei[i] = _indexes[i][idx[0]]

            failed = np.argwhere(LargerNei == -1).flatten()
            while len(failed) > 1 and _r < 10000:
                _r = _r * 10
                _indexes, _dists = posTree.query_radius(pos[failed], r=_r, return_distance=True, sort_results=True)
                for i in range(len(_indexes)):
                    idx = np.argwhere(rhos[_indexes[i]] > rhos[_indexes[i][0]])
                    if idx.shape[0] == 0:
                        pass
                    else:
                        LargerNei[failed[i]] = _indexes[i][idx[0]]
                failed = np.argwhere(LargerNei == -1).flatten()

            LargerNei = LargerNei.astype(int)
            label = LargerNei * 0 - 1
            for i in range(len(centroid)):
                label[centroid[i]] = i
            decreasingsortedIdxRhos = np.argsort(-rhos)
            for i in decreasingsortedIdxRhos:
                if label[i] == -1:
                    label[i] = label[LargerNei[i]]

            val = data[8].to_numpy()
            refinedLoop = []
            label = label.flatten()
            for l in set(label):
                idx = np.argwhere(label == l).flatten()
                if len(idx) > 0:
                    refinedLoop.append(idx[np.argmax(val[idx])])

            if refine:
                loopPds.append(data.loc[refinedLoop])
            else:
                loopPds.append(data.loc[centroid])

        loopPd = pd.concat(loopPds).sort_values(6, ascending=False)
        loopPd[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].to_csv(output, sep='\t', header=False, index=False)
        loopPDP(output)


    dc = 4
    candidates = output_directory + 'beforeClustering/beforeClustering.bedpe'
    resol = 5000
    minscore = 0.5
    mindelta = 5
    output_cluster = output_directory + 'result.txt'

    refine = False
    pool(dc, candidates, resol, mindelta, minscore, output_cluster, refine)
