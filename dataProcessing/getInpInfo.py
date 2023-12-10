def run(nof_file_path, chromosome_list, low_cp_scope, high_cp_scope, output_directory):
    def count_rows(filename):
        with open(filename, 'r') as f:
            count = 0
            for line in f:
                count += 1
            return count

    a = chromosome_list

    resolution = 5000
    lower = int(low_cp_scope)
    upper = int(high_cp_scope)

    for n in range(len(a)):
        all = []
        big_matrix_name = nof_file_path + '/chr' + str(a[n]) + '-5kb.KRnorm'
        path_out_name = output_directory + '/interactionPointInformation/center_point_all_chr' + str(a[n]) + '.txt'

        lenth = count_rows(big_matrix_name)
        for i in range(1, lenth + 1):
            end1 = int(i + (lower / resolution))
            end2 = int(i + (upper / resolution))
            if end1 <= lenth and end2 <= lenth:
                for j in range(end1, end2 + 1):
                    all.append([i, j])
            elif end1 <= lenth < end2:
                for j in range(end1, lenth + 1):
                    all.append([i, j])
            else:
                break

        outfile = open(path_out_name, 'w+')
        for info in all:
            info[0] = str(info[0])
            info[1] = str(info[1])
            temp = '\t'.join(info)
            outfile.write('chr' + str(a[n]) + '\t' + temp + '\n')
        outfile.close()
