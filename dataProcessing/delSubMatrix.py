def run(chromosome_list, output_directory):
    a = chromosome_list

    for n in range(len(a)):
        center_point_path = output_directory + '/deleteInteractionPointInformation/center_point_remove_less1_chr' + str(a[n]) + '.txt'
        npy_path = output_directory + '/subMatrix/npy/KR_5kb_matrix_chr' + str(a[n]) + '_0_all.npy'
        save_center_path = output_directory + '/deleteSubMatrix/predict_chr' + str(a[n]) + '_finall.txt'
        save_npy_path = output_directory + '/deleteSubMatrix/KR_5kb_matrix_chr' + str(a[n]) + '_finall.npy'

        zero_nums = 500

        cp_all = []
        infile = open(center_point_path, 'r')
        for line in infile:
            line = line.strip('\n')
            cp_all.append(line)
        infile.close()

        cp_new = []
        array_list = []
        array_all = np.load(npy_path, mmap_mode='r')
        array_num = array_all.shape[0]
        delete_num = 0
        part = array_num // 10
        for i in range(array_num):
            zero_counts = np.count_nonzero(array_all[i][:-1] == 0)
            if zero_counts <= zero_nums:
                array_list.append(array_all[i].tolist())
                cp_new.append(cp_all[i])
            else:
                delete_num += 1

        array_new = np.array(array_list)
        np.save(save_npy_path, array_new)
        outfile = open(save_center_path, 'w+')
        for temp in cp_new:
            outfile.write(temp + '\n')
        outfile.close()
