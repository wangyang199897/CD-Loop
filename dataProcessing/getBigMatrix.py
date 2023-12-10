def run(inf_file_path, nof_file_path, chromosome_list, output_directory):
    def openreadtxt(file_name):
        with open(file_name, 'r') as file:
            data = [row.strip('\n').split('\t') for row in file]
        return data

    def readlen(file_path):
        with open(file_path, 'r') as file:
            line_count = 0
            for line in file:
                line_count += 1
        return line_count

    a = chromosome_list
    for i in range(len(a)):
        KRmatrixpath = inf_file_path + '/chr' + str(a[i]) + '-5kb.KRobserved'
        KRnormpath = nof_file_path + '/chr' + str(a[i]) + '-5kb.KRnorm'
        path_out_path = output_directory + '/bigmatrix/KR_matrix_5kb.chr' + str(a[i])

        KRmatrix = openreadtxt(KRmatrixpath)
        KRnorm = readlen(KRnormpath)
        rawbin = []
        big_matrix = np.zeros((KRnorm, KRnorm))
        for j in range(len(KRmatrix)):
            if len(KRmatrix[j]) == 3:
                lbin = int(int(KRmatrix[j][0]) / 5000)
                rbin = int(int(KRmatrix[j][1]) / 5000)
                interactions_nums = float(KRmatrix[j][2])
                if lbin == rbin:
                    big_matrix[lbin][rbin] = interactions_nums
                else:
                    big_matrix[lbin][rbin] = interactions_nums
                    big_matrix[rbin][lbin] = interactions_nums
                rawbin.append([lbin + 1, rbin + 1])
            else:
                continue

        path_out = path_out_path
        with open(path_out, 'w+') as f_out:
            for m in range(len(big_matrix)):
                for n in range(len(big_matrix[m])):
                    f_out.write(str(big_matrix[m][n]))
                    if n == len(big_matrix[m]) - 1:
                        continue
                    else:
                        f_out.write('\t')
                f_out.write('\n')
